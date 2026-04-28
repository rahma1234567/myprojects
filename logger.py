import logging
import os
from datetime import datetime

#This section creates a  date-based filenames for users
today = datetime.now().strftime("%Y-%m-%d")
SYSTEM_LOG = f"logs/system_{today}.log"
ANOMALY_LOG = f"logs/anomalies_{today}.log"


def _setup_logger(name, log_file, level=logging.INFO):
    """Create or retrieve a logger that writes to a specific file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


#There will be two separate loggers, one for the system logs and one for the anomlay logs. 
system_logger = _setup_logger("system", SYSTEM_LOG)
anomaly_logger = _setup_logger("anomaly", ANOMALY_LOG)


# Convenience functions
def log_info(message):
    """General system events (file upload, model switch, etc.)"""
    system_logger.info(message)


def log_warning(message):
    system_logger.warning(message)


def log_error(message):
    system_logger.error(message)


def log_anomaly(transaction_id, customer_id, risk_pct, model, location=None, amount=None):
    """Specifically for flagged transactions."""
    msg = f"ALERT | TxID={transaction_id} | Customer={customer_id} | Model={model} | Risk={risk_pct}%"
    if location:
        msg += f" | Location={location}"
    if amount is not None:
        msg += f" | Amount=£{amount:.2f}"
    anomaly_logger.warning(msg)