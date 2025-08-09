# Real-Time Predictive Maintenance Platform

## Overview
This project delivers an end-to-end, production-ready predictive maintenance system that turns raw sensor streams into actionable maintenance work orders. It combines deep learning, streaming analytics, and intuitive dashboards to minimize downtime, extend asset life, and maximize business value.

## Architecture at a Glance
- **src/** – core logic including synthetic data generation, model training scripts, statistical validation utilities, and shared logging helpers.
- **RealTimeProcessing/** – real-time pipeline that consumes Kafka messages, extracts features on the fly, performs model inference, and persists results.
- **IntegrationWithExistingSystems/** – adapters that interface with [openMAINT](https://www.openmaint.org/) to create work orders and to ingest asset metadata.
- **dashboards/** – configurations for Grafana and a Dash dashboard to visualize predictions, trends, and key metrics.
- **models/** – trained CNN, LSTM, hybrid CNN‑LSTM models, and supervised anomaly detection ensembles.
- **tests/** – unit and integration tests covering Kafka flows, database interactions, and the openMAINT bridge.
- **docker-compose.yml** – spins up Zookeeper and Kafka so the complete streaming stack can run locally with a single command.

## Data & Modeling Strategy
### Synthetic Data Creation & Validation
- A synthetic generator produces 10,000 samples that emulate real sensor behavior and embed four failure modes: **TWF**, **HDF**, **PWF**, and **OSF**.
- Kolmogorov–Smirnov and Chi‑Square tests compare synthetic distributions with historical data to guarantee statistical fidelity.

### Model Training
- **CNN / LSTM / CNN‑LSTM** networks trained with K‑fold cross‑validation, class balancing, early stopping, and automated hyper‑parameter tuning.
- **Isolation Forest Ensemble** that combines DBSCAN, One‑Class SVM, Local Outlier Factor, and Random Forest; predictions are weighted by validation performance.
- Detailed logs capture AUC, F1, and PR‑AUC for every fold, enabling transparent model governance.

## Real-Time Data Pipeline
1. **SensorDataSimulator** streams IoT measurements to Kafka topics.
2. **RealTimeProcessor** loads all trained models, performs feature engineering and normalization, then scores each event in real time.
3. Alerts are published to a dedicated topic where **openmaint_consumer** converts them into openMAINT work orders.
4. Metrics flow to Grafana dashboards for interactive monitoring.

## Containerised Infrastructure
- `docker-compose.yml` provisions Zookeeper and Kafka, allowing the streaming backbone to be bootstrapped in seconds.
- The same file can be extended to include PostgreSQL, Grafana, and openMAINT, ensuring a consistent and reproducible development environment.

## Getting Started
1. **Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r predictive_maintenance/requirements.txt
   ```
2. **Launch Kafka Stack**
   ```bash
   cd predictive_maintenance
   docker compose up -d
   ```
3. **Run the Full Pipeline**
   ```bash
   python run.py
   ```
   The script boots Kafka, simulates sensor data, performs inference, and interfaces with openMAINT and Grafana.

## Achievements & Challenges
- Engineered a statistically faithful synthetic data generator with complex failure logic.
- Trained multiple model architectures with dynamic configuration to avoid overfitting.
- Delivered a real-time, end-to-end pipeline—from simulation to automatic work order creation.
- Authored automated tests for Kafka infrastructure and openMAINT integration.

## Contributing
This project is built for learning and demonstration. Contributions, feedback, and collaboration are welcome!
