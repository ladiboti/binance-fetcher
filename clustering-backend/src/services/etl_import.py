import os
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

# Configuration
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

# Verify connection
with engine.connect() as connection:
    result = connection.execute(text("SELECT current_database(), current_user;"))
    for row in result:
        logger.info(f"Connected to database: {row[0]}, user: {row[1]}")

# Load data
DATA_FILE = os.path.join(os.path.dirname(__file__), "../../data/full_data_with_clusters.csv")
logger.info(f"Loading data from {DATA_FILE}")

try:
    # Load CSV file with header and low memory usage
    data = pd.read_csv(DATA_FILE, low_memory=False)
    logger.info(f"Data loaded successfully, total records: {len(data)}")
    
    # Convert timestamp to proper format
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    # Sort data by timestamp to ensure chronological processing
    data = data.sort_values(['symbol', 'timestamp'])
    
    # Extract unique symbols
    unique_symbols = data["symbol"].unique()
    logger.info(f"Number of unique currencies: {len(unique_symbols)}")
    
    # Dictionary to track current cluster for each symbol
    current_clusters = {}
    
    # Populate currencies table and get IDs
    symbol_id_map = {}
    
    with engine.begin() as connection:
        # First, alter the currencies table to add cluster_id column if it doesn't exist
        try:
            connection.execute(text("""
                ALTER TABLE currencies ADD COLUMN IF NOT EXISTS cluster_id INTEGER;
            """))
            logger.info("Added cluster_id column to currencies table if it didn't exist")
        except Exception as e:
            logger.warning(f"Could not add cluster_id column (it might already exist): {e}")
        
        for symbol in unique_symbols:
            name = symbol.replace("USDT", "")
            
            # Get initial cluster for this symbol
            initial_data = data[data['symbol'] == symbol].iloc[0]
            initial_cluster = int(initial_data['agg_cluster'])
            current_clusters[symbol] = initial_cluster
            
            # Insert or update with initial cluster
            result = connection.execute(text("""
                INSERT INTO currencies (symbol, name, cluster_id)
                VALUES (:symbol, :name, :cluster_id)
                ON CONFLICT (symbol) DO UPDATE SET
                    cluster_id = :cluster_id
                RETURNING id;
            """), {"symbol": symbol, "name": name, "cluster_id": initial_cluster})
            
            row = result.fetchone()
            if row:
                symbol_id_map[symbol] = row[0]
                logger.info(f"Inserted/Updated: {symbol} (ID: {row[0]}, Initial Cluster: {initial_cluster})")
    
    logger.info(f"Total currencies processed: {len(symbol_id_map)}")
    
    # Insert historical prices in batches
    BATCH_SIZE = 500
    total_rows = len(data)
    
    # Process data in batches
    for batch_start in range(0, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch = data.iloc[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start}-{batch_end-1} of {total_rows}")
        
        try:
            with engine.begin() as connection:
                for index, row in batch.iterrows():
                    symbol = row["symbol"]
                    currency_id = symbol_id_map.get(symbol)
                    
                    if currency_id is None:
                        logger.warning(f"Skipping: unknown symbol: {symbol}")
                        continue
                    
                    # Get the cluster ID from the row
                    new_cluster_id = int(row["agg_cluster"])
                    
                    # Check if cluster has changed
                    if new_cluster_id != current_clusters.get(symbol):
                        old_cluster_id = current_clusters.get(symbol)
                        
                        # Update the current cluster
                        current_clusters[symbol] = new_cluster_id
                        
                        # Update the currency table with the new cluster
                        connection.execute(text("""
                            UPDATE currencies 
                            SET cluster_id = :cluster_id
                            WHERE id = :currency_id
                        """), {"cluster_id": new_cluster_id, "currency_id": currency_id})
                        
                        logger.info(f"Cluster change detected for {symbol}: {old_cluster_id} -> {new_cluster_id} at {row['timestamp']}")
                    
                    # Insert data into historical_prices table
                    historical_query = text("""
                        INSERT INTO historical_prices (
                            currency_id, timestamp, cluster_id, open, high, low, close, volume,
                            sma_10, ema_10, rsi_14, bollinger_upper, bollinger_middle, bollinger_lower,
                            macd, macd_signal, atr, obv, best_bid_price, best_ask_price, best_bid_volume,
                            best_ask_volume, bid_ask_spread, order_book_imbalance, bid_volume_depth, ask_volume_depth
                        ) VALUES (
                            :currency_id, :timestamp, :cluster_id, :open, :high, :low, :close, :volume,
                            :sma_10, :ema_10, :rsi_14, :bollinger_upper, :bollinger_middle, :bollinger_lower,
                            :macd, :macd_signal, :atr, :obv, :best_bid_price, :best_ask_price, :best_bid_volume,
                            :best_ask_volume, :bid_ask_spread, :order_book_imbalance, :bid_volume_depth, :ask_volume_depth
                        )
                        ON CONFLICT DO NOTHING
                    """)
                    
                    # Insert data
                    connection.execute(historical_query, {
                        "currency_id": currency_id,
                        "timestamp": row["timestamp"],
                        "cluster_id": new_cluster_id,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "sma_10": row["sma_10"],
                        "ema_10": row["ema_10"],
                        "rsi_14": row["rsi_14"],
                        "bollinger_upper": row["bollinger_upper"],
                        "bollinger_middle": row["bollinger_middle"],
                        "bollinger_lower": row["bollinger_lower"],
                        "macd": row["macd"],
                        "macd_signal": row["macd_signal"],
                        "atr": row["atr"],
                        "obv": row["obv"],
                        "best_bid_price": row["best_bid_price"],
                        "best_ask_price": row["best_ask_price"],
                        "best_bid_volume": row["best_bid_volume"],
                        "best_ask_volume": row["best_ask_volume"],
                        "bid_ask_spread": row["bid_ask_spread"],
                        "order_book_imbalance": row["order_book_imbalance"],
                        "bid_volume_depth": row["bid_volume_depth"],
                        "ask_volume_depth": row["ask_volume_depth"]
                    })
                    
                    # Log progress
                    if (index - batch_start) % 100 == 0 and index > batch_start:
                        logger.info(f"Progress: {index - batch_start} rows inserted in current batch")
            
            logger.info(f"Batch successfully inserted: {batch_start}-{batch_end-1}")
        except Exception as e:
            logger.error(f"Error processing batch ({batch_start}-{batch_end-1}): {e}")
            logger.error("This batch was not committed due to errors")
    
    logger.info("Data import completed.")
except Exception as e:
    logger.error(f"Error loading data: {e}")
