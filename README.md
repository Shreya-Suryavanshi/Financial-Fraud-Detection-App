

# Financial Fraud Detection Application

End-to-end fraud detection platform for banking, UPI, fintech, and digital payments using:

- Python ETL pipeline
- Machine Learning (`LogisticRegression`, `RandomForest`, `XGBoost`)
- Anomaly detection (`IsolationForest`)
- Simulated real-time fraud monitoring for dashboard demo
- Streamlit dashboard (interactive analytics + prediction + reports)
- SQLite/MySQL persistence + SMTP alerting + authentication

## Features

- ETL preprocessing: missing value handling, one-hot encoding, normalization
- Model training with metric tracking and best-model selection
- Risk scoring using fraud probability + anomaly intensity
- Simulated live transaction feed for dashboard demos
- Fraud dashboard with:
  - Total transactions, fraud %, alerts
  - Pie, bar, and line charts
  - Single transaction fraud prediction
  - Risk score and fraud probability visualization
  - Search/filter fraud history and downloadable CSV reports
- Email alerts for high-risk transactions
- User authentication (default admin account)

## Project Structure

```
app/
  config.py
  database.py
  auth.py
  etl.py
  models.py
  anomaly.py
  alerts.py
  train_models.py
  realtime/
    kafka_producer.py
    spark_streaming_job.py
dashboard.py
requirements.txt
```

## Setup

1. Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

Update SMTP/MySQL values as needed. Kafka/Spark are not required for the dashboard demo.

3. Train models:

```bash
python -m app.train_models
```

4. Run dashboard:

```bash
streamlit run dashboard.py
```

Default login:

- Username: `admin`
- Password: `admin123`

## Live Simulator Demo

The dashboard includes a built-in live simulator for fraud analytics.
No Kafka or Spark setup is required to view the live analytics tab.

1. Run the dashboard:

```bash
streamlit run dashboard.py
```

2. Open the browser at `http://localhost:8501`.

3. Use the `Live Analytics` tab to see simulated real-time fraud signals.

## Security and Scalability Notes

- Switch from SQLite to MySQL using `MYSQL_URL` for production workload.
- Use strong secrets and rotate SMTP credentials.
- Add TLS/SSO and RBAC if deploying externally.
- Containerize with Docker/Kubernetes and use managed Kafka/Spark for scale.

