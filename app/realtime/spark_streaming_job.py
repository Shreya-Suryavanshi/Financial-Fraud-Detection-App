from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from app.config import settings
from app.database import FraudTransaction, SessionLocal, init_db
from app.etl import transform_preprocessor
from app.models import load_artifacts, risk_score
from app.alerts import send_alert_email


schema = StructType(
    [
        StructField("step", IntegerType()),
        StructField("type", StringType()),
        StructField("amount", DoubleType()),
        StructField("nameOrig", StringType()),
        StructField("oldbalanceOrg", DoubleType()),
        StructField("newbalanceOrig", DoubleType()),
        StructField("nameDest", StringType()),
        StructField("oldbalanceDest", DoubleType()),
        StructField("newbalanceDest", DoubleType()),
        StructField("isFraud", IntegerType()),
        StructField("isFlaggedFraud", IntegerType()),
    ]
)


def main():
    spark = (
        SparkSession.builder.appName("fraud-streaming-monitor")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    init_db()

    # Load ML artifacts on the driver (models run inside foreachBatch).
    model, feature_cols, imputer, scaler = load_artifacts(settings.artifacts_dir)

    stream_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.kafka_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_df = stream_df.select(from_json(col("value").cast("string"), schema).alias("tx")).select("tx.*")

    # Score each micro-batch and insert predictions into the DB.
    def score_and_persist(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return

        # Convert only the necessary columns for preprocessing.
        cols = [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]
        pdf = batch_df.select(*cols).toPandas()
        if pdf.empty:
            return

        processed = transform_preprocessor(pdf, imputer=imputer, scaler=scaler, feature_columns=feature_cols)

        fraud_prob = model.predict_proba(processed)[:, 1].astype(float)
        risk = risk_score(fraud_prob).astype(float)

        # Persist predictions (driver-side insert).
        now = datetime.utcnow()
        risk_threshold = settings.smtp_alert_threshold
        to_insert: list[dict[str, Any]] = []

        for i in range(len(pdf)):
            tx_type = str(pdf.loc[i, "type"])
            amount = float(pdf.loc[i, "amount"])
            to_insert.append(
                {
                    "step": int(pdf.loc[i, "step"]),
                    "tx_type": tx_type,
                    "amount": amount,
                    "oldbalance_org": float(pdf.loc[i, "oldbalanceOrg"]),
                    "newbalance_orig": float(pdf.loc[i, "newbalanceOrig"]),
                    "oldbalance_dest": float(pdf.loc[i, "oldbalanceDest"]),
                    "newbalance_dest": float(pdf.loc[i, "newbalanceDest"]),
                    "is_fraud": bool(fraud_prob[i] >= 0.5),
                    "fraud_probability": float(fraud_prob[i]),
                    "anomaly_score": float(anomaly_score[i]),
                    "risk_score": float(risk[i]),
                    # created_at defaults to datetime.utcnow() in SQLAlchemy,
                    # but we still insert a consistent timestamp for this batch.
                    "created_at": now,
                }
            )

        session = SessionLocal()
        try:
            session.bulk_insert_mappings(FraudTransaction, to_insert)
            session.commit()
        finally:
            session.close()

        # Email alerts for high-risk transactions (best-effort; failures do not stop the stream).
        if settings.smtp_host and settings.smtp_user and settings.smtp_password:
            high_risk_idx = np.where(risk >= risk_threshold)[0]
            if len(high_risk_idx) > 0:
                # Avoid one email per record in streaming demos: summarize up to 5.
                top = high_risk_idx[:5]
                alert_body = [
                    f"Batch {batch_id} detected {len(high_risk_idx)} high-risk transactions (threshold={risk_threshold}).",
                    f"Generated at: {now.isoformat()} UTC",
                    "",
                    "Top transactions:",
                ]
                for i in top:
                    alert_body.append(
                        f"- Risk {risk[i]:.1f}/100 | Fraud Prob {fraud_prob[i]:.2%} | Type {pdf.loc[i,'type']} | Amount {pdf.loc[i,'amount']:.2f}"
                    )
                sent = send_alert_email(
                    subject="High Risk Fraud Alert (Live Streaming)",
                    body="\n".join(alert_body),
                )

                # Mark alerted rows best-effort by setting alert_sent=true for those rows in this batch.
                # For simplicity, we mark by created_at + risk threshold window.
                if sent:
                    session2 = SessionLocal()
                    try:
                        end_time = now + timedelta(seconds=1)
                        session2.query(FraudTransaction).filter(
                            FraudTransaction.created_at >= now,
                            FraudTransaction.created_at < end_time,
                            FraudTransaction.risk_score >= risk_threshold,
                            FraudTransaction.alert_sent == False,  # noqa: E712
                        ).update({FraudTransaction.alert_sent: True}, synchronize_session=False)
                        session2.commit()
                    finally:
                        session2.close()

    checkpoint = settings.spark_checkpoint_dir.as_posix()

    query = (
        parsed_df.writeStream.outputMode("append")
        .foreachBatch(score_and_persist)
        .option("checkpointLocation", checkpoint)
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()

