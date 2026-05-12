from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable, KafkaTimeoutError

from app.config import settings


def create_producer():
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def stream_transactions(delay_seconds: float = 0.3, nrows: int = 5000):
    try:
        producer = create_producer()
    except NoBrokersAvailable as e:
        print(
            "Kafka broker not reachable."
            f" bootstrap_servers={settings.kafka_bootstrap_servers}"
        )
        print("Start Kafka broker first, or update KAFKA_BOOTSTRAP_SERVERS in your .env.")
        raise e
    df = pd.read_csv(settings.data_file, nrows=nrows)
    for _, row in df.iterrows():
        payload = row.to_dict()
        try:
            producer.send(settings.kafka_topic, payload)
        except KafkaTimeoutError:
            print("Message send timed out. Check broker/topic connectivity.")
            raise
        time.sleep(delay_seconds)
    producer.flush()
    producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream sample transactions into Kafka")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between messages (seconds)")
    parser.add_argument("--nrows", type=int, default=500, help="Number of rows to stream from CSV")
    args = parser.parse_args()
    stream_transactions(delay_seconds=args.delay, nrows=args.nrows)

