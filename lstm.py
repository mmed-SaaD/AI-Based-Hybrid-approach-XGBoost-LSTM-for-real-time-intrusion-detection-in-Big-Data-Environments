import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, RepeatVector, 
                                         TimeDistributed, BatchNormalization, 
                                         Input, Bidirectional, Attention)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras import regularizers
    print("✅ TensorFlow/Keras imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    exit(1)

class ImprovedLSTMAnomalyDetector:
    def __init__(self, sequence_length=10, contamination=0.10):
        self.sequence_length = sequence_length
        self.contamination = contamination
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.threshold = None
        self.feature_columns = None
        self.mse_mean = None
        self.mse_std = None
        
    def extract_doubt_traffic_from_xgboost(self, raw_data_path, xgboost_model_path):
        print("="*60)
        print("EXTRACTING DOUBT TRAFFIC FROM RAW DATA")
        print("="*60)
        
        raw_data = pd.read_csv(raw_data_path)
        xgboost_package = joblib.load(xgboost_model_path)
        
        print(f"Loaded raw data: {raw_data.shape}")
        
        scaled_data = xgboost_package['scaler'].transform(raw_data)
        predictions = xgboost_package['model'].predict(scaled_data)
        probabilities = xgboost_package['model'].predict_proba(scaled_data)
        
        predicted_classes = xgboost_package['label_encoder'].inverse_transform(predictions)
        
        doubt_indices = []
        confidence_threshold = 0.65
        
        for i, (pred_class, proba) in enumerate(zip(predicted_classes, probabilities)):
            max_confidence = np.max(proba)
            
            if (pred_class == 'DOUBT' or 
                (pred_class == 'BENIGN' and max_confidence < confidence_threshold)):
                doubt_indices.append(i)
        
        doubt_traffic = raw_data.iloc[doubt_indices].copy()
        
        print(f"Found {len(doubt_traffic)} doubt samples ({len(doubt_traffic)/len(raw_data)*100:.2f}%)")
        
        doubt_traffic.to_csv('raw_doubt_traffic.csv', index=False)
        print("💾 Raw doubt traffic saved")
        
        return doubt_traffic
    
    def smart_clean_data(self, data):
        print("🔧 Smart data cleaning and feature engineering...")
        
        cleaned_data = data.copy()
        
        # First phase: Replace infinities
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        
        # Second phase: Remove constant and near-constant columns
        for col in cleaned_data.columns:
            if cleaned_data[col].nunique() <= 1:
                cleaned_data = cleaned_data.drop(columns=[col])
            elif cleaned_data[col].std() < 1e-6:
                cleaned_data = cleaned_data.drop(columns=[col])
        
        # Third phase: Smart outlier handling using IQR
        for col in cleaned_data.columns:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # --------->        Fill NaN with median
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
        
        # Fourth phase: Remove highly correlated features (reduce redundancy)
        corr_matrix = cleaned_data.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                   if any(upper_triangle[column] > 0.98)]
        
        if len(to_drop) > 0:
            print(f"   Removing {len(to_drop)} highly correlated features")
            cleaned_data = cleaned_data.drop(columns=to_drop)
        
        # Fifth phase: Feature importance-based selection (variance threshold)
        variances = cleaned_data.var()
        low_var_cols = variances[variances < 0.01].index.tolist()
        if len(low_var_cols) > 0:
            print(f"   Removing {len(low_var_cols)} low variance features")
            cleaned_data = cleaned_data.drop(columns=low_var_cols)
        
        # Final phase : Final validation
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        cleaned_data = cleaned_data.fillna(cleaned_data.median())
        
        print(f"✅ Data cleaned")
        print(f"   Final shape: {cleaned_data.shape}")
        print(f"   Features: {cleaned_data.shape[1]}")
        print(f"   Range: [{cleaned_data.values.min():.2f}, {cleaned_data.values.max():.2f}]")
        
        return cleaned_data
        
    def load_doubt_traffic(self, raw_data_path=None, xgboost_model_path=None, doubt_csv_path=None):
        print("="*60)
        print("LOADING DOUBT TRAFFIC DATA")
        print("="*60)
        
        if doubt_csv_path and pd.io.common.file_exists(doubt_csv_path):
            self.doubt_data = pd.read_csv(doubt_csv_path)
            print(f"Loaded existing doubt traffic: {self.doubt_data.shape}")
        elif raw_data_path and xgboost_model_path:
            self.doubt_data = self.extract_doubt_traffic_from_xgboost(raw_data_path, xgboost_model_path)
        else:
            raise ValueError("Must provide data source")
        
        self.doubt_data = self.smart_clean_data(self.doubt_data)
        self.feature_columns = self.doubt_data.columns.tolist()
        
        return self.doubt_data
    
    def create_sequences(self, data, stride=1):
        sequences = []
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Not enough data: {len(data)} samples")
        
        # Overlapping windows for better learning
        for i in range(0, len(data) - self.sequence_length + 1, stride):
            sequences.append(data[i:(i + self.sequence_length)])
        
        sequences = np.array(sequences)
        print(f"✅ Created {len(sequences)} sequences with stride {stride}")
        
        return sequences
    
    def build_improved_lstm_autoencoder(self, input_shape):
        """Build improved LSTM autoencoder with attention"""
        print("="*60)
        print("BUILDING IMPROVED LSTM AUTOENCODER")
        print("="*60)
        print(f"Input shape: {input_shape}")
        
        # Encoder
        inputs = Input(shape=input_shape)
        
        # Bidirectional LSTM 
        x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.001)))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(32, activation='tanh', return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Bottleneck with stronger compression
        encoded = LSTM(16, activation='tanh', return_sequences=False,
                      kernel_regularizer=regularizers.l2(0.001))(x)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.2)(encoded)
        
        # Decoder
        x = RepeatVector(input_shape[0])(encoded)
        
        x = LSTM(16, activation='tanh', return_sequences=True,
                kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(LSTM(32, activation='tanh', return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.001)))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output
        outputs = TimeDistributed(Dense(input_shape[1]))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mse', 'mae']
        )
        
        print(model.summary())
        
        return model
    
    def prepare_data(self, test_size=0.2, val_size=0.15):
        print("="*60)
        print("PREPARING DATA")
        print("="*60)
        
        data_array = self.doubt_data.values
        print(f"Original data shape: {data_array.shape}")
        
        # Two-stage scaling: RobustScaler + MinMaxScaler
        scaled_data = self.scaler.fit_transform(data_array)
        scaled_data = self.feature_scaler.fit_transform(scaled_data)
        
        print(f"Scaled data range: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
        print(f"Scaled data mean: {scaled_data.mean():.3f}")
        print(f"Scaled data std: {scaled_data.std():.3f}")
        
        # Create sequences with stride=3 for training (more samples)
        train_sequences = self.create_sequences(scaled_data, stride=3)
        
        # Split
        X_temp, X_test = train_test_split(train_sequences, test_size=test_size, 
                                          random_state=42, shuffle=True)
        X_train, X_val = train_test_split(X_temp, test_size=val_size/(1-test_size), 
                                          random_state=42, shuffle=True)
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Validation sequences: {X_val.shape}")
        print(f"Test sequences: {X_test.shape}")
        
        return X_train, X_val, X_test, train_sequences
    
    def train_model(self, X_train, X_val, epochs=150, batch_size=64):
        print("="*60)
        print("TRAINING IMPROVED LSTM AUTOENCODER")
        print("="*60)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_improved_lstm_autoencoder(input_shape)
        
        # Improved callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        )
        
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        print("\n🚀 Starting training...")
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_scheduler, checkpoint],
            verbose=1
        )
        
        # Load best model
        self.model.load_weights('best_lstm_model.h5')
        
        # Evaluate
        train_pred = self.model.predict(X_train, verbose=0)
        val_pred = self.model.predict(X_val, verbose=0)
        
        train_mse = np.mean((X_train - train_pred)**2)
        val_mse = np.mean((X_val - val_pred)**2)
        
        train_mae = np.mean(np.abs(X_train - train_pred))
        val_mae = np.mean(np.abs(X_val - val_pred))
        
        print(f"\n✅ Training completed!")
        print(f"📊 Final Metrics:")
        print(f"   Training MSE: {train_mse:.6f} | MAE: {train_mae:.6f}")
        print(f"   Validation MSE: {val_mse:.6f} | MAE: {val_mae:.6f}")
        print(f"   Overfitting ratio: {train_mse/val_mse:.3f}")
        
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (Huber)')
        axes[0].set_title('Model Loss During Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # MSE
        axes[1].plot(history.history['mse'], label='Training MSE', linewidth=2)
        axes[1].plot(history.history['val_mse'], label='Validation MSE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('MSE During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MAE
        axes[2].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[2].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MAE')
        axes[2].set_title('MAE During Training')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_lstm_training_history.png', dpi=300, bbox_inches='tight')
        print("💾 Training history saved")
        plt.show()
    
    def calculate_dynamic_threshold(self, X_val):
        print("="*60)
        print("CALCULATING DYNAMIC THRESHOLD")
        print("="*60)
        
        reconstructions = self.model.predict(X_val, verbose=0)
        
        # Calculate multiple error metrics
        mse = np.mean((X_val - reconstructions)**2, axis=(1, 2))
        mae = np.mean(np.abs(X_val - reconstructions), axis=(1, 2))
        
        # Store statistics for dynamic thresholding
        self.mse_mean = np.mean(mse)
        self.mse_std = np.std(mse)
        
        print(f"MSE statistics:")
        print(f"   Mean: {self.mse_mean:.6f}")
        print(f"   Std: {self.mse_std:.6f}")
        print(f"   Median: {np.median(mse):.6f}")
        print(f"   Q1: {np.percentile(mse, 25):.6f}")
        print(f"   Q3: {np.percentile(mse, 75):.6f}")
        
        # Method 1: Statistical threshold (mean + k*std)
        k = 2.5  # Tunable parameter
        threshold_statistical = self.mse_mean + k * self.mse_std
        
        # Method 2: Percentile-based
        threshold_percentile = (1 - self.contamination) * 100
        threshold_perc = np.percentile(mse, threshold_percentile)
        
        # Method 3: IQR method
        Q1 = np.percentile(mse, 25)
        Q3 = np.percentile(mse, 75)
        IQR = Q3 - Q1
        threshold_iqr = Q3 + 1.5 * IQR
        
        # Use the median of all methods for robustness
        self.threshold = np.median([threshold_statistical, threshold_perc, threshold_iqr])
        
        print(f"\nThreshold candidates:")
        print(f"   Statistical (μ + {k}σ): {threshold_statistical:.6f}")
        print(f"   Percentile ({threshold_percentile:.1f}%): {threshold_perc:.6f}")
        print(f"   IQR method: {threshold_iqr:.6f}")
        print(f"   SELECTED (median): {self.threshold:.6f}")
        
        # Estimate expected anomaly rate
        expected_rate = (mse > self.threshold).mean()
        print(f"   Expected anomaly rate: {expected_rate*100:.2f}%")
        
        self.plot_threshold_analysis(mse, mae)
        
        return self.threshold
    
    def plot_threshold_analysis(self, mse, mae):
        """Enhanced threshold visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # MSE Histogram
        axes[0, 0].hist(mse, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 0].axvline(self.threshold, color='red', linestyle='--', linewidth=2,
                          label=f'Threshold: {self.threshold:.4f}')
        axes[0, 0].axvline(self.mse_mean, color='green', linestyle='--', linewidth=1,
                          label=f'Mean: {self.mse_mean:.4f}')
        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reconstruction Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE Timeline
        axes[0, 1].plot(mse, alpha=0.7, color='steelblue')
        axes[0, 1].axhline(self.threshold, color='red', linestyle='--', linewidth=2,
                          label='Threshold')
        axes[0, 1].fill_between(range(len(mse)), 0, self.threshold, 
                                alpha=0.2, color='green', label='Normal')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('MSE Over Samples')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(mse, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot (Normality Check)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # MAE Histogram
        axes[1, 0].hist(mae, bins=50, alpha=0.7, edgecolor='black', color='coral')
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('MAE Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter: MSE vs MAE
        anomalies = mse > self.threshold
        axes[1, 1].scatter(mse[~anomalies], mae[~anomalies], 
                          alpha=0.6, s=20, c='green', label='Normal')
        axes[1, 1].scatter(mse[anomalies], mae[anomalies], 
                          alpha=0.6, s=20, c='red', label='Anomaly')
        axes[1, 1].axvline(self.threshold, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('MSE')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('MSE vs MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Box plot comparison
        if anomalies.sum() > 0 and (~anomalies).sum() > 0:
            bp = axes[1, 2].boxplot([mse[~anomalies], mse[anomalies]], 
                                   labels=['Normal', 'Anomaly'],
                                   patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            axes[1, 2].set_ylabel('MSE')
            axes[1, 2].set_title('MSE by Classification')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_lstm_threshold_analysis.png', dpi=300, bbox_inches='tight')
        print("💾 Threshold analysis saved")
        plt.show()
    
    def detect_anomalies(self, sequences):
        """Detect anomalies with confidence scores"""
        print("="*60)
        print("DETECTING ANOMALIES")
        print("="*60)
        
        reconstructions = self.model.predict(sequences, verbose=0)
        mse = np.mean((sequences - reconstructions)**2, axis=(1, 2))
        
        # Binary classification
        anomalies = mse > self.threshold
        
        # Confidence scores (normalized distance from threshold)
        confidence_scores = np.abs(mse - self.threshold) / self.mse_std
        
        print(f"Results:")
        print(f"   Total samples: {len(sequences)}")
        print(f"   Anomalies: {anomalies.sum()} ({anomalies.mean()*100:.2f}%)")
        print(f"   Normal: {(~anomalies).sum()} ({(~anomalies).mean()*100:.2f}%)")
        print(f"   MSE range: [{mse.min():.6f}, {mse.max():.6f}]")
        print(f"   Avg confidence: {confidence_scores.mean():.3f}")
        
        return anomalies, mse, confidence_scores
    
    def save_model(self):
        """Save model and metadata"""
        print("="*60)
        print("SAVING MODEL")
        print("="*60)
        
        self.model.save('improved_lstm_anomaly_detector.h5')
        print("💾 Model saved: improved_lstm_anomaly_detector.h5")
        
        metadata = {
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'threshold': self.threshold,
            'mse_mean': self.mse_mean,
            'mse_std': self.mse_std,
            'sequence_length': self.sequence_length,
            'contamination': self.contamination,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(metadata, 'improved_lstm_metadata.joblib')
        print("💾 Metadata saved: improved_lstm_metadata.joblib")
    
    def run_improved_pipeline(self, raw_data_path=None, xgboost_model_path=None, doubt_csv_path=None):
        """Run complete improved training pipeline"""
        print("\n" + "="*60)
        print("STARTING IMPROVED LSTM TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        self.load_doubt_traffic(raw_data_path, xgboost_model_path, doubt_csv_path)
        
        # Prepare data
        X_train, X_val, X_test, all_sequences = self.prepare_data()
        
        # Train
        history = self.train_model(X_train, X_val, epochs=150, batch_size=64)
        
        # Set threshold
        self.calculate_dynamic_threshold(X_val)
        
        # Detect anomalies
        anomalies, mse, confidence = self.detect_anomalies(X_test)
        
        # Save
        self.save_model()
        
        print("\n" + "="*60)
        print("✅ IMPROVED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return anomalies, mse, confidence, X_test

def main():
    """Main function"""
    
    raw_data_path = "test_features.csv"
    xgboost_model_path = "xgboost_ids_grouped.joblib"
    
    detector = ImprovedLSTMAnomalyDetector(
        sequence_length=10,
        contamination=0.10
    )
    
    try:
        anomalies, mse, confidence, X_test = detector.run_improved_pipeline(
            raw_data_path=raw_data_path,
            xgboost_model_path=xgboost_model_path
        )
        
        print(f"\n✅ Training successful!")
        print(f"   Detected {anomalies.sum()} anomalies")
        print(f"   Anomaly rate: {anomalies.mean()*100:.2f}%")
        print(f"   MSE range: [{mse.min():.6f}, {mse.max():.6f}]")
        print(f"   Avg confidence: {confidence.mean():.3f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
