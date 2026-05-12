from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_file: Path = base_dir / "PS_20174392719_1491204439457_log.csv"
    artifacts_dir: Path = base_dir / "artifacts"
    reports_dir: Path = base_dir / "reports"
    sqlite_url: str = os.getenv("SQLITE_URL", f"sqlite:///{(base_dir / 'fraud_app.db').as_posix()}")
    mysql_url: str | None = os.getenv("MYSQL_URL")
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    alert_from: str = os.getenv("ALERT_FROM", "fraud-monitor@example.com")
    alert_to: str = os.getenv("ALERT_TO", "admin@example.com")
    smtp_alert_threshold: float = float(os.getenv("SMTP_ALERT_THRESHOLD", "80"))
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "transactions")
    spark_checkpoint_dir: Path = artifacts_dir / "spark_checkpoint"


settings = Settings()
settings.artifacts_dir.mkdir(exist_ok=True, parents=True)
settings.reports_dir.mkdir(exist_ok=True, parents=True)
settings.spark_checkpoint_dir.mkdir(exist_ok=True, parents=True)

