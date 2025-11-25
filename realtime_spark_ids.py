import pandas as pd
import numpy as np
import joblib
import time
import sys
import warnings
import argparse
import json
from collections import deque, defaultdict
from colorama import Fore, Back, Style, init

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Spark and Kafka imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    SPARK_AVAILABLE = True
    print(f"{Fore.GREEN}✅ PySpark loaded successfully")
except ImportError:
    SPARK_AVAILABLE = False
    print(f"{Fore.RED}❌ PySpark not available. Please install pyspark.")

# LSTM model imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    LSTM_AVAILABLE = True
    print(f"{Fore.GREEN}✅ TensorFlow/Keras loaded successfully")
except ImportError:
    LSTM_AVAILABLE = False
    print(f"{Fore.YELLOW}⚠️  TensorFlow not available. LSTM analysis will be skipped.")

class KafkaStreamingLSTMInference:
    def __init__(self, 
                 kafka_brokers='192.168.100.15:9092',
                 kafka_topic='netflow.events',
                 xgboost_model_path='xgboost_ids_grouped.joblib', 
                 lstm_model_path='lstm_raw_data_anomaly_detector.h5',
                 lstm_metadata_path='lstm_raw_data_metadata.joblib'):
        """Initialize the Kafka streaming LSTM inference system"""
        self.kafka_brokers = kafka_brokers
        self.kafka_topic = kafka_topic
        self.xgboost_model_path = xgboost_model_path
        self.lstm_model_path = lstm_model_path
        self.lstm_metadata_path = lstm_metadata_path
        
        # Model components
        self.xgboost_package = None
        self.lstm_model = None
        self.lstm_metadata = None
        self.spark = None
        
        # LSTM inference improvements (same as original)
        self.lstm_scaler_fitted_data = None
        self.recent_samples_buffer = deque(maxlen=100)  # Rolling buffer for sequence creation
        
        # Separated statistics by device
        self.device_stats = defaultdict(lambda: {
            'total_processed': 0,
            'benign_count': 0,
            'attack_count': 0,
            'doubt_count': 0,
            'lstm_analyzed': 0,
            'lstm_confirmed_anomaly': 0,
            'lstm_confirmed_benign': 0,
            'lstm_errors': 0,
            'attack_types': {}
        })
        
        # Overall combined statistics for compatibility
        self.stats = {
            'total_processed': 0,
            'benign_count': 0,
            'attack_count': 0,
            'doubt_count': 0,
            'lstm_analyzed': 0,
            'lstm_confirmed_anomaly': 0,
            'lstm_confirmed_benign': 0,
            'lstm_errors': 0,
            'attack_types': {}
        }
        
        # Feature columns for CSV parsing (you may need to adjust based on your CSV structure)
        self.feature_columns = None
        
    def initialize_spark(self):
        """Initialize Spark session with Kafka support"""
        if not SPARK_AVAILABLE:
            print(f"{Fore.RED}❌ Spark not available")
            sys.exit(1)
            
        try:
            print(f"{Fore.CYAN}🔧 Initializing Spark session with Kafka support...")
            
            self.spark = SparkSession.builder \
                .appName("KafkaStreamingLSTMInference") \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
                .getOrCreate()
                
            # Set log level to reduce Spark noise
            self.spark.sparkContext.setLogLevel("WARN")
            
            print(f"{Fore.GREEN}✅ Spark session initialized successfully!")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error initializing Spark: {e}")
            return False
    
    def load_xgboost_model(self):
        """Load the trained XGBoost model and preprocessors (same as original)"""
        try:
            print(f"{Fore.CYAN}🔧 Loading XGBoost IDS model...")
            self.xgboost_package = joblib.load(self.xgboost_model_path)
            
            # Verify model components
            required_components = ['model', 'scaler', 'label_encoder', 'feature_columns']
            for component in required_components:
                if component not in self.xgboost_package:
                    raise KeyError(f"Missing component: {component}")
            
            # Store feature columns for CSV parsing
            self.feature_columns = self.xgboost_package['feature_columns']
            
            print(f"{Fore.GREEN}✅ XGBoost model loaded successfully!")
            print(f"{Fore.CYAN}   - Features: {len(self.xgboost_package['feature_columns'])}")
            print(f"{Fore.CYAN}   - Classes: {list(self.xgboost_package['label_encoder'].classes_)}")
            
        except FileNotFoundError:
            print(f"{Fore.RED}❌ XGBoost model file not found: {self.xgboost_model_path}")
            sys.exit(1)
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading XGBoost model: {e}")
            sys.exit(1)
    
    def load_lstm_model(self):
        """Load the trained LSTM model and metadata (same as original)"""
        if not LSTM_AVAILABLE:
            print(f"{Fore.YELLOW}⚠️  Skipping LSTM model loading (TensorFlow not available)")
            return False
            
        try:
            print(f"{Fore.CYAN}🔧 Loading new LSTM anomaly detection model...")
            print(f"{Fore.CYAN}   Model path: {self.lstm_model_path}")
            print(f"{Fore.CYAN}   Metadata path: {self.lstm_metadata_path}")
            
            # Load LSTM model with custom objects to handle potential issues
            self.lstm_model = load_model(self.lstm_model_path, compile=False)
            
            # Recompile with stable settings
            self.lstm_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='mse',
                metrics=['mae']
            )
            
            print(f"{Fore.GREEN}✅ New LSTM model loaded and recompiled successfully!")
            
            # Load metadata
            self.lstm_metadata = joblib.load(self.lstm_metadata_path)
            print(f"{Fore.GREEN}✅ New LSTM metadata loaded successfully!")
            print(f"{Fore.CYAN}   - Model type: {self.lstm_metadata.get('model_type', 'standard')}")
            print(f"{Fore.CYAN}   - Sequence length: {self.lstm_metadata['sequence_length']}")
            print(f"{Fore.CYAN}   - Anomaly threshold: {self.lstm_metadata['threshold']:.6f}")
            print(f"{Fore.CYAN}   - Features: {len(self.lstm_metadata.get('feature_columns', []))}")
            
            # Validate threshold
            if self.lstm_metadata['threshold'] <= 0 or np.isnan(self.lstm_metadata['threshold']):
                print(f"{Fore.YELLOW}⚠️  Invalid threshold detected, using default: 0.01")
                self.lstm_metadata['threshold'] = 0.01
            
            # Create reference scaling data from model metadata if available
            if 'scaler' in self.lstm_metadata and hasattr(self.lstm_metadata['scaler'], 'mean_'):
                # Generate synthetic reference data based on scaler statistics
                n_features = len(self.lstm_metadata['scaler'].mean_)
                self.lstm_scaler_fitted_data = np.random.normal(
                    self.lstm_metadata['scaler'].mean_,
                    self.lstm_metadata['scaler'].scale_,
                    size=(100, n_features)
                )
                print(f"{Fore.CYAN}📊 LSTM scaling reference prepared with synthetic data")
            
            return True
            
        except FileNotFoundError as e:
            print(f"{Fore.YELLOW}⚠️  New LSTM model files not found: {e}")
            return False
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Error loading new LSTM model: {e}")
            return False
    
    def safe_scale_sample(self, sample):
        """Safely scale a single sample for LSTM input (same as original)"""
        try:
            # Convert to numpy array if needed
            if isinstance(sample, pd.Series):
                sample = sample.values
            sample = np.array(sample).reshape(1, -1)
            
            # Handle infinite or NaN values
            sample = np.nan_to_num(sample, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Apply scaling with context
            if self.lstm_scaler_fitted_data is not None:
                context_data = np.vstack([self.lstm_scaler_fitted_data[:10], sample])
                scaled_context = self.lstm_metadata['scaler'].transform(context_data)
                scaled_sample = scaled_context[-1:]
            else:
                scaled_sample = self.lstm_metadata['scaler'].transform(sample)
            
            # Clip values to prevent extreme scaling
            scaled_sample = np.clip(scaled_sample, -5, 5)
            
            return scaled_sample[0]
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error in scaling: {e}")
            return np.zeros(sample.shape[1] if len(sample.shape) > 1 else len(sample))
    
    def create_stable_lstm_sequence(self, current_sample):
        """Create LSTM sequence with numerical stability (same as original)"""
        sequence_length = self.lstm_metadata['sequence_length']
        
        # Add current sample to buffer
        self.recent_samples_buffer.append(current_sample)
        
        # Create sequence
        if len(self.recent_samples_buffer) >= sequence_length:
            sequence_data = np.array(list(self.recent_samples_buffer)[-sequence_length:])
        else:
            needed = sequence_length - len(self.recent_samples_buffer)
            padding = [current_sample] * needed
            sequence_data = np.array(padding + list(self.recent_samples_buffer))
        
        # Ensure numerical stability
        sequence_data = np.nan_to_num(sequence_data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return sequence_data.reshape(1, sequence_length, -1)
    
    def analyze_doubt_with_lstm_fixed(self, doubt_sample, sample_id, device_id):
        """Fixed LSTM analysis with numerical stability (same as original with device tracking)"""
        if self.lstm_model is None or self.lstm_metadata is None:
            return None, None
        
        try:
            # Scale the sample safely
            scaled_sample = self.safe_scale_sample(doubt_sample)
            
            # Create stable sequence
            lstm_input = self.create_stable_lstm_sequence(scaled_sample)
            
            # Validate input
            if np.any(np.isnan(lstm_input)) or np.any(np.isinf(lstm_input)):
                print(f"{Fore.YELLOW}⚠️  Invalid LSTM input detected, skipping analysis")
                self.device_stats[device_id]['lstm_errors'] += 1
                self.stats['lstm_errors'] += 1
                return None, None
            
            # Get LSTM reconstruction with error handling
            try:
                reconstruction = self.lstm_model.predict(lstm_input, verbose=0)
                
                if np.any(np.isnan(reconstruction)) or np.any(np.isinf(reconstruction)):
                    print(f"{Fore.YELLOW}⚠️  Invalid LSTM reconstruction, skipping analysis")
                    self.device_stats[device_id]['lstm_errors'] += 1
                    self.stats['lstm_errors'] += 1
                    return None, None
                    
            except Exception as pred_error:
                print(f"{Fore.YELLOW}⚠️  LSTM prediction error: {pred_error}")
                self.device_stats[device_id]['lstm_errors'] += 1
                self.stats['lstm_errors'] += 1
                return None, None
            
            # Calculate reconstruction error safely
            diff = lstm_input - reconstruction
            mse = np.mean(np.power(diff, 2))
            
            # Validate MSE
            if np.isnan(mse) or np.isinf(mse) or mse < 0:
                print(f"{Fore.YELLOW}⚠️  Invalid MSE calculated: {mse}, using fallback")
                self.device_stats[device_id]['lstm_errors'] += 1
                self.stats['lstm_errors'] += 1
                return None, None
            
            # Clamp MSE to reasonable range
            mse = np.clip(mse, 1e-10, 1e6)
            
            # Compare with threshold
            threshold = self.lstm_metadata['threshold']
            is_anomaly = mse > threshold
            
            # Calculate anomaly score
            raw_score = mse / threshold
            if raw_score > 10:
                anomaly_score = 10 + np.log10(raw_score - 9)
                anomaly_score = min(anomaly_score, 100)
            else:
                anomaly_score = raw_score
            
            # Update statistics for both device and overall
            self.device_stats[device_id]['lstm_analyzed'] += 1
            self.stats['lstm_analyzed'] += 1
            
            if is_anomaly:
                self.device_stats[device_id]['lstm_confirmed_anomaly'] += 1
                self.stats['lstm_confirmed_anomaly'] += 1
                result = 'ANOMALY'
            else:
                self.device_stats[device_id]['lstm_confirmed_benign'] += 1
                self.stats['lstm_confirmed_benign'] += 1
                result = 'BENIGN'
            
            return result, {
                'reconstruction_error': float(mse),
                'threshold': float(threshold),
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'model_type': self.lstm_metadata.get('model_type', 'standard')
            }
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error in new LSTM analysis: {e}")
            self.device_stats[device_id]['lstm_errors'] += 1
            self.stats['lstm_errors'] += 1
            return None, None
    
    def classify_traffic_xgboost(self, sample_data, confidence_threshold=0.7):
        """Classify traffic using XGBoost (same as original)"""
        # Scale the sample
        sample_scaled = self.xgboost_package['scaler'].transform([sample_data])
        
        # Get prediction and probabilities
        prediction = self.xgboost_package['model'].predict(sample_scaled)[0]
        probabilities = self.xgboost_package['model'].predict_proba(sample_scaled)[0]
        
        # Convert prediction back to class name
        predicted_class = self.xgboost_package['label_encoder'].inverse_transform([prediction])[0]
        max_confidence = np.max(probabilities)
        
        # Get class names
        class_names = self.xgboost_package['label_encoder'].classes_
        
        # Determine traffic type
        if predicted_class == 'DOUBT':
            return 'DOUBT', predicted_class, max_confidence
            
        elif predicted_class == 'BENIGN':
            benign_confidence = max_confidence
            
            # Calculate attack probability
            attack_classes = [i for i, name in enumerate(class_names) 
                            if name not in ['BENIGN', 'DOUBT']]
            
            if attack_classes:
                attack_proba = np.sum(probabilities[attack_classes])
                
                if (benign_confidence < confidence_threshold or 
                    (attack_proba > 0.2 and benign_confidence < 0.9)):
                    return 'DOUBT', 'Suspicious BENIGN', max_confidence
            
            return 'BENIGN', predicted_class, max_confidence
            
        else:
            return 'ATTACK', predicted_class, max_confidence
    
    def display_result(self, traffic_type, predicted_class, confidence, sample_id, device_id, lstm_result=None):
        """Display result with device ID (same formatting as original)"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        if traffic_type == 'BENIGN':
            print(f"{Fore.GREEN}[{timestamp}] ✅ Sample #{sample_id:04d} ({device_id}) | Benign Traffic | Confidence: {confidence:.3f}")
            
        elif traffic_type == 'ATTACK':
            print(f"{Fore.RED}{Style.BRIGHT}[{timestamp}] 🚨 Sample #{sample_id:04d} ({device_id}) | ALERT: {predicted_class} attack attempt detected! | Confidence: {confidence:.3f}")
            
        elif traffic_type == 'DOUBT':
            print(f"{Fore.YELLOW}[{timestamp}] ⚠️  Sample #{sample_id:04d} ({device_id}) | INFO: Suspicious traffic detected, forwarding to NEW LSTM model for deeper analysis...")
            
            # Show LSTM analysis result
            if lstm_result and lstm_result[0]:
                lstm_decision, lstm_details = lstm_result
                model_type = lstm_details.get('model_type', 'standard')
                if lstm_decision == 'ANOMALY':
                    print(f"{Fore.RED}[{timestamp}] 🔍 Sample #{sample_id:04d} ({device_id}) | NEW LSTM ANALYSIS ({model_type}): Confirmed ANOMALY! | Error: {lstm_details['reconstruction_error']:.8f} (threshold: {lstm_details['threshold']:.8f}) | Score: {lstm_details['anomaly_score']:.3f}")
                else:
                    print(f"{Fore.GREEN}[{timestamp}] 🔍 Sample #{sample_id:04d} ({device_id}) | NEW LSTM ANALYSIS ({model_type}): Confirmed BENIGN | Error: {lstm_details['reconstruction_error']:.8f} (threshold: {lstm_details['threshold']:.8f}) | Score: {lstm_details['anomaly_score']:.3f}")
            elif lstm_result and lstm_result[0] is None:
                print(f"{Fore.CYAN}[{timestamp}] 🔍 Sample #{sample_id:04d} ({device_id}) | NEW LSTM ANALYSIS: Analysis failed (numerical instability)")
            else:
                print(f"{Fore.CYAN}[{timestamp}] 🔍 Sample #{sample_id:04d} ({device_id}) | NEW LSTM ANALYSIS: Not available (model not loaded)")
    
    def update_stats(self, traffic_type, predicted_class, device_id, lstm_result=None):
        """Update processing statistics for both device-specific and overall"""
        # Update device-specific stats
        self.device_stats[device_id]['total_processed'] += 1
        
        # Update overall stats
        self.stats['total_processed'] += 1
        
        if traffic_type == 'BENIGN':
            self.device_stats[device_id]['benign_count'] += 1
            self.stats['benign_count'] += 1
        elif traffic_type == 'ATTACK':
            self.device_stats[device_id]['attack_count'] += 1
            self.stats['attack_count'] += 1
            
            # Update attack types for device
            if predicted_class in self.device_stats[device_id]['attack_types']:
                self.device_stats[device_id]['attack_types'][predicted_class] += 1
            else:
                self.device_stats[device_id]['attack_types'][predicted_class] = 1
            
            # Update overall attack types
            if predicted_class in self.stats['attack_types']:
                self.stats['attack_types'][predicted_class] += 1
            else:
                self.stats['attack_types'][predicted_class] = 1
                
        elif traffic_type == 'DOUBT':
            self.device_stats[device_id]['doubt_count'] += 1
            self.stats['doubt_count'] += 1
    
    def display_device_stats(self, device_id, device_stats):
        """Display statistics for a specific device"""
        total = device_stats['total_processed']
        if total == 0:
            return
            
        print(f"{Fore.CYAN}📱 {device_id.upper()} STATISTICS:")
        print(f"{Fore.WHITE}Total Processed: {total}")
        print(f"{Fore.GREEN}Benign Traffic: {device_stats['benign_count']} ({device_stats['benign_count']/total*100:.1f}%)")
        print(f"{Fore.RED}Attack Traffic: {device_stats['attack_count']} ({device_stats['attack_count']/total*100:.1f}%)")
        print(f"{Fore.YELLOW}Suspicious Traffic: {device_stats['doubt_count']} ({device_stats['doubt_count']/total*100:.1f}%)")
        
        # LSTM statistics for this device
        if self.lstm_model is not None and device_stats['lstm_analyzed'] > 0:
            model_type = self.lstm_metadata.get('model_type', 'standard')
            print(f"\n{Fore.MAGENTA}🧠 {device_id.upper()} LSTM DEEP ANALYSIS ({model_type}):")
            print(f"{Fore.MAGENTA}   Total Analyzed: {device_stats['lstm_analyzed']}")
            print(f"{Fore.RED}   Confirmed Anomalies: {device_stats['lstm_confirmed_anomaly']}")
            print(f"{Fore.GREEN}   Confirmed Benign: {device_stats['lstm_confirmed_benign']}")
            print(f"{Fore.YELLOW}   Analysis Errors: {device_stats['lstm_errors']}")
            
            if device_stats['lstm_analyzed'] > 0:
                anomaly_rate = device_stats['lstm_confirmed_anomaly'] / device_stats['lstm_analyzed'] * 100
                success_rate = (device_stats['lstm_analyzed'] / (device_stats['lstm_analyzed'] + device_stats['lstm_errors'])) * 100
                print(f"{Fore.MAGENTA}   Anomaly Rate: {anomaly_rate:.1f}%")
                print(f"{Fore.MAGENTA}   Success Rate: {success_rate:.1f}%")
        
        if device_stats['attack_types']:
            print(f"\n{Fore.RED}🚨 {device_id.upper()} Attack Types Detected:")
            for attack_type, count in device_stats['attack_types'].items():
                print(f"{Fore.RED}   - {attack_type}: {count} times")
    
    def display_stats(self):
        """Display separated statistics by device and overall summary"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}📊 DEVICE-SEPARATED STATISTICS")
        print(f"{Fore.CYAN}{'='*70}")
        
        # Display stats for each device that has processed data
        active_devices = [device_id for device_id, stats in self.device_stats.items() if stats['total_processed'] > 0]
        
        if not active_devices:
            print(f"{Fore.YELLOW}No data processed yet...")
            print(f"{Fore.CYAN}{'='*70}\n")
            return
        
        # Sort devices for consistent display order
        for device_id in sorted(active_devices):
            device_stats = self.device_stats[device_id]
            self.display_device_stats(device_id, device_stats)
            print(f"{Fore.CYAN}{'-'*50}")
        
        # Display overall combined statistics
        total = self.stats['total_processed']
        if total > 0:
            print(f"\n{Fore.CYAN}🌐 COMBINED OVERALL STATISTICS:")
            print(f"{Fore.WHITE}Total Processed: {total}")
            print(f"{Fore.GREEN}Benign Traffic: {self.stats['benign_count']} ({self.stats['benign_count']/total*100:.1f}%)")
            print(f"{Fore.RED}Attack Traffic: {self.stats['attack_count']} ({self.stats['attack_count']/total*100:.1f}%)")
            print(f"{Fore.YELLOW}Suspicious Traffic: {self.stats['doubt_count']} ({self.stats['doubt_count']/total*100:.1f}%)")
            
            # Overall LSTM statistics
            if self.lstm_model is not None and self.stats['lstm_analyzed'] > 0:
                model_type = self.lstm_metadata.get('model_type', 'standard')
                print(f"\n{Fore.MAGENTA}🧠 COMBINED LSTM DEEP ANALYSIS ({model_type}):")
                print(f"{Fore.MAGENTA}   Total Analyzed: {self.stats['lstm_analyzed']}")
                print(f"{Fore.RED}   Confirmed Anomalies: {self.stats['lstm_confirmed_anomaly']}")
                print(f"{Fore.GREEN}   Confirmed Benign: {self.stats['lstm_confirmed_benign']}")
                print(f"{Fore.YELLOW}   Analysis Errors: {self.stats['lstm_errors']}")
                
                if self.stats['lstm_analyzed'] > 0:
                    anomaly_rate = self.stats['lstm_confirmed_anomaly'] / self.stats['lstm_analyzed'] * 100
                    success_rate = (self.stats['lstm_analyzed'] / (self.stats['lstm_analyzed'] + self.stats['lstm_errors'])) * 100
                    print(f"{Fore.MAGENTA}   Anomaly Rate: {anomaly_rate:.1f}%")
                    print(f"{Fore.MAGENTA}   Success Rate: {success_rate:.1f}%")
            
            if self.stats['attack_types']:
                print(f"\n{Fore.RED}🚨 Combined Attack Types Detected:")
                for attack_type, count in self.stats['attack_types'].items():
                    print(f"{Fore.RED}   - {attack_type}: {count} times")
        
        print(f"{Fore.CYAN}{'='*70}\n")
    
    def parse_kafka_message(self, kafka_value, device_id):
        """Parse Kafka message into feature array"""
        try:
            # Split CSV line into values
            values = kafka_value.split(',')
            
            # Convert to float array
            feature_array = np.array([float(v) for v in values])
            
            return feature_array
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error parsing Kafka message from {device_id}: {e}")
            return None
    
    def process_kafka_stream(self, show_stats_every=25):
        """Process Kafka stream with Spark Streaming"""
        try:
            print(f"\n{Fore.CYAN}🚀 Starting Kafka Streaming IDS Monitoring with Device Separation...")
            print(f"{Fore.CYAN}📡 Kafka brokers: {self.kafka_brokers}")
            print(f"{Fore.CYAN}📋 Topic: {self.kafka_topic}")
            print(f"{Fore.CYAN}🔄 Supporting multiple devices: node 1, node 2, etc.")
            print(f"{Fore.CYAN}📊 Stats update every {show_stats_every} samples (separated by device)")
            
            if self.lstm_model is not None:
                model_type = self.lstm_metadata.get('model_type', 'standard')
                print(f"{Fore.MAGENTA}🧠 NEW LSTM deep analysis: ENABLED (Type: {model_type})")
            else:
                print(f"{Fore.YELLOW}🧠 NEW LSTM deep analysis: DISABLED")
                
            print(f"{Fore.CYAN}⏹️  Press Ctrl+C to stop")
            print(f"{Fore.GREEN}🎯 Ready to process traffic with separated device statistics...")
            print(f"{Fore.GREEN}📨 Waiting for Kafka messages...\n")
            
            # Create streaming DataFrame
            kafka_df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.kafka_brokers) \
                .option("subscribe", self.kafka_topic) \
                .option("startingOffsets", "latest") \
                .load()
            
            # Convert Kafka messages to string
            messages_df = kafka_df.select(
                col("key").cast("string").alias("device_id"),
                col("value").cast("string").alias("csv_data"),
                col("timestamp")
            )
            
            # Process each batch
            def process_batch(batch_df, batch_id):
                if batch_df.count() > 0:
                    messages = batch_df.collect()
                    
                    for row in messages:
                        device_id = row.device_id if row.device_id else "unknown"
                        csv_data = row.csv_data
                        
                        # Parse CSV data
                        feature_array = self.parse_kafka_message(csv_data, device_id)
                        
                        if feature_array is not None:
                            sample_id = self.stats['total_processed'] + 1
                            
                            # Process through XGBoost and LSTM
                            traffic_type, predicted_class, confidence = self.classify_traffic_xgboost(feature_array)
                            
                            lstm_result = None
                            if traffic_type == 'DOUBT':
                                lstm_result = self.analyze_doubt_with_lstm_fixed(feature_array, sample_id, device_id)
                            
                            # Display result
                            self.display_result(traffic_type, predicted_class, confidence, sample_id, device_id, lstm_result)
                            
                            # Update statistics (both device-specific and overall)
                            self.update_stats(traffic_type, predicted_class, device_id, lstm_result)
                            
                            # Show stats periodically
                            if self.stats['total_processed'] % show_stats_every == 0:
                                self.display_stats()
            
            # Start the streaming query
            query = messages_df.writeStream \
                .foreachBatch(process_batch) \
                .outputMode("append") \
                .trigger(processingTime='1 seconds') \
                .start()
            
            # Wait for termination
            query.awaitTermination()
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⏹️  Kafka streaming stopped by user")
        except Exception as e:
            print(f"{Fore.RED}❌ Error in Kafka streaming: {e}")
        finally:
            # Final statistics
            self.display_stats()
            print(f"{Fore.GREEN}✅ Kafka streaming inference completed!")
            
            if self.spark:
                self.spark.stop()

def main():
    """Main function with Kafka streaming"""
    parser = argparse.ArgumentParser(description='Kafka Streaming LSTM IDS Inference with Separated Device Statistics')
    parser.add_argument('--brokers', default='192.168.100.15:9092', 
                       help='Kafka brokers (default: 192.168.100.15:9092)')
    parser.add_argument('--topic', default='netflow.events', 
                       help='Kafka topic (default: netflow.events)')
    parser.add_argument('--xgboost-model', default='xgboost_ids_grouped.joblib',
                       help='XGBoost model path')
    parser.add_argument('--lstm-model', default='lstm_raw_data_anomaly_detector.h5',
                       help='LSTM model path')
    parser.add_argument('--lstm-metadata', default='lstm_raw_data_metadata.joblib',
                       help='LSTM metadata path')
    parser.add_argument('--stats-interval', type=int, default=25,
                       help='Show stats every N samples')
    
    args = parser.parse_args()
    
    print(f"{Fore.MAGENTA}{Style.BRIGHT}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║      KAFKA STREAMING XGBoost + LSTM IDS Real-Time Inference         ║")
    print("║           Multi-Level Network Security Monitor                       ║")
    print("║         (Spark Streaming + Kafka + Device Separation)               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)
    
    # Initialize inference system
    ids = KafkaStreamingLSTMInference(
        kafka_brokers=args.brokers,
        kafka_topic=args.topic,
        xgboost_model_path=args.xgboost_model,
        lstm_model_path=args.lstm_model,
        lstm_metadata_path=args.lstm_metadata
    )
    
    # Initialize Spark
    if not ids.initialize_spark():
        sys.exit(1)
    
    # Load models
    ids.load_xgboost_model()
    lstm_loaded = ids.load_lstm_model()
    
    if lstm_loaded:
        print(f"\033[32m✅ Both XGBoost and NEW LSTM models loaded successfully!\033[0m")
        print(f"\033[32m✅ NEW LSTM model uses raw data format (consistent with training)\033[0m")
        print(f"\033[36m🔄 Processing flow: Kafka Stream → XGBoost → DOUBT detection → NEW LSTM analysis\033[0m")
        print(f"\033[36m📊 Statistics will be separated by device (node 1, node 2, etc.)\033[0m")
    else:
        print(f"\033[34m✅ XGBoost model loaded. NEW LSTM analysis disabled.\033[0m")
        print(f"\033[36m📊 Statistics will be separated by device (node 1, node 2, etc.)\033[0m")
    
    # Start Kafka streaming
    try:
        ids.process_kafka_stream(show_stats_every=args.stats_interval)
    except KeyboardInterrupt:
        print(f"\n\033[33mProgram interrupted by user\033[0m")
    except Exception as e:
        print(f"\033[31mError: {e}\033[0m")

if __name__ == "__main__":
    main()
