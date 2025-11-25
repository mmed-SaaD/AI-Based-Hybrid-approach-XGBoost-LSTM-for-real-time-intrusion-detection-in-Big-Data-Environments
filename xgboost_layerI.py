
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Try different XGBoost imports
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    print("✅ XGBoost imported successfully")
except ImportError as e:
    print(f"❌ XGBoost import failed: {e}")
    print("Please install XGBoost: pip install xgboost")
    exit(1)

class CompleteXGBoostIDS:
    def __init__(self, csv_path):
        """Initialize with path to CIC-IDS2017 CSV file"""
        self.csv_path = csv_path
        self.data = None
        self.processed_data = {}
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def group_attack_categories(self, labels):
        """Group related attacks into broader categories"""
        grouped_labels = labels.copy()
        
        # Group related attacks into one global class
        attack_groups = {
            # DoS attacks
            'DoS': ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'DDoS'],
            
            # Web attacks
            'Web Attack': ['Web Attack – Brute Force', 'Web Attack – XSS', 'Web Attack – Sql Injection',
                          'Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection'],
            
            # Brute force attacks
            'Brute Force': ['FTP-Patator', 'SSH-Patator'],
            
            # Reconnaissance
            'Reconnaissance': ['PortScan'],
            
            # Malware/Bot
            'Malware': ['Bot'],
            
            # Advanced attacks
            'Advanced Attack': ['Infiltration', 'Heartbleed']
        }
        
        for group_name, attack_types in attack_groups.items():
            for attack_type in attack_types:
                grouped_labels = grouped_labels.replace(attack_type, group_name)
        
        return grouped_labels
        
    def load_and_clean_data(self):
        print("="*60)
        print("LOADING AND CLEANING DATA")
        print("="*60)
        
        print("Loading CIC-IDS2017 dataset...")
        self.data = pd.read_csv(self.csv_path)
        print(f"Original shape: {self.data.shape}")
        
        # Quick check to display original label distribution
        print("\nOriginal label distribution:")
        print(self.data['Label'].value_counts())
        
        # Applying attack grouping
        print("\nGrouping related attacks...")
        self.data['Label'] = self.group_attack_categories(self.data['Label'])
        
        print("\nGrouped label distribution:")
        print(self.data['Label'].value_counts())
        
        # Remove duplicates
        initial_rows = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_rows - self.data.shape[0]} duplicate rows")
        
        # Handle infinite values
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        
        # Identify and remove non-numeric columns 
        non_feature_cols = ['Label']
        for col in self.data.columns:
            if col not in non_feature_cols and self.data[col].dtype == 'object':
                print(f"Removing non-numeric column: {col}")
                non_feature_cols.append(col)
        
        feature_cols = [col for col in self.data.columns if col not in non_feature_cols]
        X = self.data[feature_cols].copy()
        y = self.data['Label'].copy()
        
        # Fill missing values with median
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Remove constant columns
        constant_cols = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"Removing constant columns: {constant_cols}")
            X = X.drop(columns=constant_cols)
        
        # Remove highly correlated features
        print("Removing highly correlated features...")
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_triangle.columns 
                            if any(upper_triangle[column] > 0.95)]
        
        if high_corr_features:
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
        
        self.feature_columns = X.columns.tolist()
        self.data = pd.concat([X, y], axis=1)
        
        print(f"✅ Final cleaned shape: {self.data.shape}")
        print(f"✅ Final feature count: {len(self.feature_columns)}")
        
    def create_doubt_class_and_split(self, doubt_ratio=0.1):
        print("="*60)
        print("CREATING DOUBT CLASS AND SPLITTING DATA")
        print("="*60)
        
        X = self.data[self.feature_columns]
        y = self.data['Label']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.05, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.263, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Create doubt class from training BENIGN samples
        print(f"\nCreating doubt class ({doubt_ratio*100}% of BENIGN)...")
        benign_indices = np.where(y_train == 'BENIGN')[0]
        np.random.seed(42)
        doubt_indices = np.random.choice(
            benign_indices, 
            size=int(len(benign_indices) * doubt_ratio), 
            replace=False
        )
        
        y_train_with_doubt = y_train.copy()
        y_train_with_doubt.iloc[doubt_indices] = 'DOUBT'
        
        print(f"Created {len(doubt_indices)} doubt samples")
        print("\nUpdated training class distribution:")
        print(y_train_with_doubt.value_counts())
        
        # Save test data without labels - This can later be used during the inference tests, and we will use the seperated labels for validation
        X_test.to_csv('test_features.csv', index=False)
        test_label_counts = y_test.value_counts()
        label_df = pd.DataFrame({
            'class': test_label_counts.index,
            'occurrences_number': test_label_counts.values
        })
        label_df.to_csv('test_labels.csv', index=False)
        print("✅ Test data saved to test_features.csv and test_labels.csv")
        
        self.processed_data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train_with_doubt, 'y_val': y_val, 'y_test': y_test
        }
        
    def handle_imbalance_and_scale(self):
        print("="*60)
        print("HANDLING CLASS IMBALANCE AND SCALING")
        print("="*60)
        
        X_train = self.processed_data['X_train']
        y_train = self.processed_data['y_train']
        
        print("Original class distribution:")
        print(y_train.value_counts())
        
        # Get unique classes and their counts
        class_counts = y_train.value_counts()
        
        # Create sampling strategy for undersampling - Handling minorities
        undersample_strategy = {}
        if 'BENIGN' in class_counts:
            undersample_strategy['BENIGN'] = min(500000, class_counts['BENIGN'])
        
        # Create sampling strategy for oversampling  - Handling Majorities
        oversample_strategy = {}
        for class_name, count in class_counts.items():
            if count < 1000 and class_name not in ['BENIGN', 'DOUBT']:
                oversample_strategy[class_name] = min(2000, count * 3)
        
        # Apply resampling if needed
        if undersample_strategy:
            undersampler = RandomUnderSampler(
                sampling_strategy=undersample_strategy,
                random_state=42
            )
            X_train, y_train = undersampler.fit_resample(X_train, y_train)
            print(f"Applied undersampling: {undersample_strategy}")
        
        if oversample_strategy:
            # Use minimum neighbors available
            min_samples = min([y_train.value_counts()[cls] for cls in oversample_strategy.keys()])
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            smote = SMOTE(
                sampling_strategy=oversample_strategy,
                random_state=42,
                k_neighbors=k_neighbors
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"Applied SMOTE: {oversample_strategy}")
        
        print("Resampled class distribution:")
        print(pd.Series(y_train).value_counts())
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_columns
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(self.processed_data['X_val']),
            columns=self.feature_columns
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.processed_data['X_test']),
            columns=self.feature_columns
        )
        
        # Encode labels
        all_labels = np.concatenate([
            y_train, 
            self.processed_data['y_val'], 
            self.processed_data['y_test']
        ])
        self.label_encoder.fit(all_labels)
        
        self.processed_data.update({
            'X_train_final': X_train_scaled,
            'X_val_final': X_val_scaled,
            'X_test_final': X_test_scaled,
            'y_train_final': self.label_encoder.transform(y_train),
            'y_val_final': self.label_encoder.transform(self.processed_data['y_val']),
            'y_test_final': self.label_encoder.transform(self.processed_data['y_test'])
        })
        
        print("✅ Data preparation completed!")
        
    def train_xgboost(self):
        """Train XGBoost model"""
        print("="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # Initialize XGBoost with explicit class
        try:
            self.model = XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False,
                n_jobs=-1
            )
        except Exception as e:
            print(f"❌ Error creating XGBClassifier: {e}")
            print("Trying alternative XGBoost initialization...")
            
            # Alternative initialization
            self.model = xgb.sklearn.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        # Train with early stopping
        print("🚀 Training XGBoost...")
        self.model.fit(
            self.processed_data['X_train_final'],
            self.processed_data['y_train_final'],
            eval_set=[
                (self.processed_data['X_train_final'], self.processed_data['y_train_final']),
                (self.processed_data['X_val_final'], self.processed_data['y_val_final'])
            ],
            verbose=100
        )
        
        print("✅ XGBoost training completed!")
        
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("="*60)
        print("EVALUATING MODEL PERFORMANCE")
        print("="*60)
        
        # Predictions on validation set
        y_pred = self.model.predict(self.processed_data['X_val_final'])
        y_val_true = self.processed_data['y_val_final']
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_true, y_pred)
        precision = precision_score(y_val_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val_true, y_pred, average='weighted', zero_division=0)
        
        print(f"📊 VALIDATION RESULTS:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Detailed classification report
        class_names = self.label_encoder.classes_
        print(f"\n📋 Classification Report:")
        print(classification_report(y_val_true, y_pred, target_names=class_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_val_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - XGBoost IDS (Grouped Categories)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_grouped.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.yticks(range(len(top_features)), [self.feature_columns[i] for i in top_features])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance_grouped.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def identify_doubt_traffic(self, confidence_threshold=0.7):
        """Identify doubt traffic for GNN+AE analysis"""
        print("="*60)
        print("IDENTIFYING DOUBT TRAFFIC")
        print("="*60)
        
        X_val = self.processed_data['X_val_final']
        probabilities = self.model.predict_proba(X_val)
        predictions = self.model.predict(X_val)
        
        # Find doubt cases
        max_proba = np.max(probabilities, axis=1)
        low_confidence_mask = max_proba < confidence_threshold
        
        # DOUBT class predictions
        class_names = self.label_encoder.classes_
        doubt_class_mask = np.zeros(len(predictions), dtype=bool)
        if 'DOUBT' in class_names:
            doubt_class_id = np.where(class_names == 'DOUBT')[0][0]
            doubt_class_mask = (predictions == doubt_class_id)
        
        # Suspicious BENIGN
        suspicious_benign_mask = np.zeros(len(predictions), dtype=bool)
        if 'BENIGN' in class_names:
            benign_class_id = np.where(class_names == 'BENIGN')[0][0]
            benign_predictions = (predictions == benign_class_id)
            benign_proba = probabilities[:, benign_class_id]
            
            # Calculate attack probability (sum of non-BENIGN, non-DOUBT classes)
            attack_classes = [i for i, name in enumerate(class_names) 
                            if name not in ['BENIGN', 'DOUBT']]
            if attack_classes:
                attack_proba = np.sum(probabilities[:, attack_classes], axis=1)
                suspicious_benign_mask = (benign_predictions & 
                                        (attack_proba > 0.2) & 
                                        (benign_proba < 0.9))
        
        # Combine all doubt criteria
        final_doubt_mask = low_confidence_mask | doubt_class_mask | suspicious_benign_mask
        doubt_indices = np.where(final_doubt_mask)[0]
        
        print(f"🔍 Doubt Traffic Analysis:")
        print(f"   Low confidence: {np.sum(low_confidence_mask)} samples")
        print(f"   DOUBT class: {np.sum(doubt_class_mask)} samples")
        print(f"   Suspicious BENIGN: {np.sum(suspicious_benign_mask)} samples")
        print(f"   📊 TOTAL DOUBT: {len(doubt_indices)} samples ({len(doubt_indices)/len(X_val)*100:.2f}%)")
        
        # Save doubt traffic for GNN+AE
        doubt_features = X_val.iloc[doubt_indices]
        doubt_features.to_csv('doubt_traffic_grouped.csv', index=False)
        print("💾 Doubt traffic saved to doubt_traffic_grouped.csv")
        
        return doubt_indices, doubt_features
        
    def inference_on_test_csv(self):
        """Run inference on the saved test CSV"""
        print("="*60)
        print("INFERENCE ON TEST SET")
        print("="*60)
        
        # Load test features
        X_test_csv = pd.read_csv('test_features.csv')
        print(f"Loaded test data: {X_test_csv.shape}")
        
        # Scale test features
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_csv),
            columns=self.feature_columns
        )
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        # Convert predictions back to class names
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'prediction': predicted_classes,
            'max_probability': np.max(probabilities, axis=1)
        })
        
        # Add class probabilities
        for i, class_name in enumerate(self.label_encoder.classes_):
            results_df[f'prob_{class_name}'] = probabilities[:, i]
        
        # Save results
        results_df.to_csv('test_predictions_grouped.csv', index=False)
        print("💾 Test predictions saved to test_predictions_grouped.csv")
        
        # Show prediction summary
        print("📊 Prediction Summary:")
        print(pd.Series(predicted_classes).value_counts())
        
    def save_model(self):
        """Save the trained model and preprocessors"""
        print("="*60)
        print("SAVING MODEL")
        print("="*60)
        
        # Save complete model package
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_package, 'xgboost_ids_grouped.joblib')
        print("💾 Complete model saved to xgboost_ids_grouped.joblib")
        
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 STARTING COMPLETE XGBOOST IDS PIPELINE WITH GROUPED CATEGORIES")
        print("🚀 This will take several minutes...")
        
        # Run all steps
        self.load_and_clean_data()
        self.create_doubt_class_and_split()
        self.handle_imbalance_and_scale()
        self.train_xgboost()
        self.evaluate_model()
        self.identify_doubt_traffic()
        self.inference_on_test_csv()
        self.save_model()
        
        print("="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("📁 Files created:")
        print("   - test_features.csv (test set without labels)")
        print("   - test_labels.csv (test labels with counts)")
        print("   - doubt_traffic_grouped.csv (suspicious samples for GNN+AE)")
        print("   - test_predictions_grouped.csv (final predictions)")
        print("   - xgboost_ids_grouped.joblib (trained model)")
        print("   - confusion_matrix_grouped.png")
        print("   - feature_importance_grouped.png")

def main():
    csv_path = "/home/kali/Desktop/eden/dataset/MachineLearningCVE/merged_cic_ids2017.csv"
    ids = CompleteXGBoostIDS(csv_path)
    ids.run_complete_pipeline()

if __name__ == "__main__":
    main()
