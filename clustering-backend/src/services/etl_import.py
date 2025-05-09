import os
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
POSTGRES_URI = os.getenv("POSTGRES_URI")
logger.info(f"Connecting to PostgreSQL database at {POSTGRES_URI}...")
engine = create_engine(POSTGRES_URI)

DATA_FILE = os.path.join(os.path.dirname(__file__), "../../data/full_data_with_clusters.csv")
logger.info(f"Loading data from {DATA_FILE}")
data = pd.read_csv(DATA_FILE)
unique_symbols = data["symbol"].unique()
logger.info(f"Data loaded successfully, total records: {len(data)}")

logger.info("Inserting currencies into 'currencies' table...")
inserted_count = 0
with engine.connect() as connection:
    for symbol in unique_symbols:
        result = connection.execute(text("""
            INSERT INTO currencies (symbol, name)
            VALUES (:symbol, :name)
            ON CONFLICT (symbol) DO NOTHING
            RETURNING id;
        """), {"symbol": symbol, "name": symbol})

        # Count successful inserts
        if result.fetchone() is not None:
            inserted_count += 1

logger.info(f"Inserted {inserted_count} unique currencies.")

logger.info("Data import completed.")
