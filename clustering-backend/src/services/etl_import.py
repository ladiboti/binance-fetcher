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
    # Explicitly specify format for to_datetime function
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    # Extract unique symbols
    unique_symbols = data["symbol"].unique()
    logger.info(f"Number of unique currencies: {len(unique_symbols)}")
    
    # Populate currencies table and get IDs
    symbol_id_map = {}
    
    with engine.begin() as connection:
        for symbol in unique_symbols:
            name = symbol.replace("USDT", "")
            
            # Insert or get existing ID
            result = connection.execute(text("""
                INSERT INTO currencies (symbol, name)
                VALUES (:symbol, :name)
                ON CONFLICT (symbol) DO NOTHING
                RETURNING id;
            """), {"symbol": symbol, "name": name})
            
            row = result.fetchone()
            if row:
                symbol_id_map[symbol] = row[0]
                logger.info(f"Inserted: {symbol} (ID: {row[0]})")
            
            # If we didn't get an ID back (because it already existed), retrieve it
            if symbol not in symbol_id_map:
                result = connection.execute(text("""
                    SELECT id FROM currencies WHERE symbol = :symbol
                """), {"symbol": symbol})
                row = result.fetchone()
                if row:
                    symbol_id_map[symbol] = row[0]
                    logger.info(f"Existing currency: {symbol} (ID: {row[0]})")
    
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
                    # Get currency ID
                    currency_id = symbol_id_map.get(row["symbol"])
                    
                    if currency_id is None:
                        logger.warning(f"Skipping: unknown symbol: {row['symbol']}")
                        continue
                    
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
                        "cluster_id": row["agg_cluster"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "sma_10": row["sma_10"] if not pd.isna(row["sma_10"]) else None,
                        "ema_10": row["ema_10"] if not pd.isna(row["ema_10"]) else None,
                        "rsi_14": row["rsi_14"] if not pd.isna(row["rsi_14"]) else None,
                        "bollinger_upper": row["bollinger_upper"] if not pd.isna(row["bollinger_upper"]) else None,
                        "bollinger_middle": row["bollinger_middle"] if not pd.isna(row["bollinger_middle"]) else None,
                        "bollinger_lower": row["bollinger_lower"] if not pd.isna(row["bollinger_lower"]) else None,
                        "macd": row["macd"] if not pd.isna(row["macd"]) else None,
                        "macd_signal": row["macd_signal"] if not pd.isna(row["macd_signal"]) else None,
                        "atr": row["atr"] if not pd.isna(row["atr"]) else None,
                        "obv": row["obv"] if not pd.isna(row["obv"]) else None,
                        "best_bid_price": row["best_bid_price"] if not pd.isna(row["best_bid_price"]) else None,
                        "best_ask_price": row["best_ask_price"] if not pd.isna(row["best_ask_price"]) else None,
                        "best_bid_volume": row["best_bid_volume"] if not pd.isna(row["best_bid_volume"]) else None,
                        "best_ask_volume": row["best_ask_volume"] if not pd.isna(row["best_ask_volume"]) else None,
                        "bid_ask_spread": row["bid_ask_spread"] if not pd.isna(row["bid_ask_spread"]) else None,
                        "order_book_imbalance": row["order_book_imbalance"] if not pd.isna(row["order_book_imbalance"]) else None,
                        "bid_volume_depth": row["bid_volume_depth"] if not pd.isna(row["bid_volume_depth"]) else None,
                        "ask_volume_depth": row["ask_volume_depth"] if not pd.isna(row["ask_volume_depth"]) else None
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
