import os
import json
import argparse
import pymongo
import logging
from datetime import datetime, timedelta
from binance.client import Client
from services import get_historical_klines
from dotenv import load_dotenv

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["crypto-db"] 
collection = db["klines"]  

trading_pairs = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "BNBUSDT",   # Binance Coin
    "SOLUSDT",   # Solana
    "XRPUSDT",   # Ripple
    "DOGEUSDT",  # Dogecoin
    "TONUSDT",   # Toncoin
    "ADAUSDT",   # Cardano
    "AVAXUSDT",  # Avalanche
    "DOTUSDT",   # Polkadot
    "MATICUSDT", # Polygon
    "SHIBUSDT",  # Shiba Inu
    "TRXUSDT",   # TRON
    "LTCUSDT",   # Litecoin
    "APTUSDT",   # Aptos
    "INJUSDT",   # Injective
    "ARBUSDT",   # Arbitrum
    "OPUSDT",    # Optimism
    "PEPEUSDT",  # PEPE
    "RNDRUSDT"   # Render Token
]

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Binance Klines and save to JSON.")

    parser.add_argument("--interval", type=str, default=Client.KLINE_INTERVAL_1HOUR, help="Candle interval (e.g., 1m, 5m, 1h, 1d).")
    parser.add_argument("--days", type=int, default=30, help="Number of past days to fetch.")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of candles per request (max: 1000).")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Compute the start time based on the number of days
    start_time = datetime.now() - timedelta(days=args.days)

    for pair in trading_pairs:
        logger.info(f"Fetching data for {pair}...")

        try:
            # Fetch historical Klines
            data = get_historical_klines(
                symbol=pair,
                interval=args.interval,
                start_time=start_time,
                limit=args.limit
            )

            # Check if trading pair is listed
            if not data or len(data[0]) < 14:
                logger.warning(f"Pair not listed on Binance or no data available for {pair}")
                continue
            
            logger.info(f"Retrieved {len(data)} candles for {pair} ({args.interval}) over the past {args.days} days.")
            # logger.debug(f"Sample candle for {pair}: {data[0]}")

            # Prepare JSON output
            output = {
                "symbol": pair,
                "interval": args.interval,
                "days": args.days,
                "limit": args.limit,
                "data": data
            }

            # Generate a descriptive filename
            filename = f"{pair}_{args.interval}.json"
            file_path = os.path.join(DATA_DIR, filename)

            # Save the data to JSON
            try:
                with open(file_path, "w") as json_file:
                    json.dump(output, json_file, indent=4)
                logger.info(f"Data saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save {file_path}: {e}")
                continue

            # Save to MongoDB
            try:
                collection.delete_many({"symbol": pair})
                collection.insert_one(output)
                logger.info(f"Data successfully committed to MongoDB for {pair}")
            except Exception as e:
                logger.error(f"MongoDB error for {pair}: {e}")

        except IndexError as e:
            logger.error(f"Pair not listed on Binance or invalid response for {pair}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {pair}: {e}")