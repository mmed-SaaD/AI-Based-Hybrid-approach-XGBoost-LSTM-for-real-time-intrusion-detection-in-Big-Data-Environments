# Real-Time Hybrid Intrusion Detection System (IDS)

**XGBoost Layer I + LSTM Layer II + Spark Streaming Pipeline**

This repository implements a real-time, hybrid AI-based Intrusion Detection System combining:

- **Layer I — XGBoost**: Multi-class classification of known attacks
- **Layer II — LSTM**: Anomaly detection for suspicious traffic
- **Spark Streaming + Kafka**: Real-time distributed ingestion & detection pipeline

The system is designed for high-volume network traffic, using flow-based features from CIC-IDS2017 & UNSW-NB15–style datasets.

## Project Structure
```
├── LSTM.py                  # LSTM anomaly detection model (Layer II)
├── XGBoost_layerI.py        # XGBoost multiclass classifier (Layer I)
├── real_time_spark_ids.py   # Full real-time IDS pipeline using Spark Streaming
└── README.md
```

## Features

### Hybrid IDS Architecture
- XGBoost for classifying known attacks
- LSTM Autoencoder for anomaly detection + zero-day alerting

### Real-Time Big Data Pipeline
- Kafka producers generate live network flows
- Spark Streaming consumer processes batches instantly
- Supports tens of thousands of flows per second

### Scalable & Distributed
- Designed for multi-node Spark clusters
- Fault-tolerant Kafka ingestion

## Models Overview

### Layer I — XGBoost
- Multi-class detection: BENIGN, DoS, Brute Force, Reconnaissance, Web Attacks, Malware, etc.
- **Outputs:**
  - Benign / Malicious / Doubtful (for Layer II review)

### Layer II — LSTM
- Trained only on benign traffic
- Detects deviations → flags anomalies as possible zero-day attacks
- **Outputs:**
  - Normal / Anomaly

## Installation & Requirements

To install all dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries

#### Core Python Libraries
```
pandas
numpy
time
sys
warnings
json
argparse
collections
joblib
```

#### Visualization
```
matplotlib
seaborn
```

#### Scikit-Learn & Data Processing
```
scikit-learn
```

Includes:
- RobustScaler
- MinMaxScaler
- StandardScaler
- LabelEncoder
- train_test_split
- classification metrics
- confusion matrix
- roc_auc_score

#### Imbalanced Learning
```
imblearn
```

Contains:
- SMOTE
- RandomUnderSampler
- ImbPipeline

#### Deep Learning
```
tensorflow
keras
```

*(Used inside LSTM.py if your script calls them — add if needed)*

#### Color Output
```
colorama
```

*(For the console spark IDS script)*

#### Spark & Kafka (Real-Time Pipeline)
```
pyspark
kafka-python
```

## How to Run the Real-Time IDS

### 1. Start Kafka Services
```bash
zookeeper-server-start.sh config/zookeeper.properties
kafka-server-start.sh config/server.properties
```

### 2. Create Kafka Topic
```bash
kafka-topics.sh --create --topic ids_traffic --bootstrap-server localhost:9092
```

### 3. Start Traffic Producers (Simulated Network Traffic)
```bash
python producer1.py
python producer2.py
```

### 4. Run Real-Time IDS (Spark Streaming)
```bash
python real_time_spark_ids.py
```

### 5. Output

- Live classification of every flow
- Global statistics summary
- Anomaly alerts from the LSTM layer

## Contact

For questions or collaboration:

- 📧 Email: your-email@example.com
- 🔗 LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)

## License

This project is open source and available under the [MIT License](LICENSE).
