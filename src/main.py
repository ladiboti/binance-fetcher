import os
import json
import argparse
import pymongo
from datetime import datetime, timedelta
from binance.client import Client
from services import get_historical_klines
from dotenv import load_dotenv

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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
    """
    Parse command-line arguments for Klines fetching.
    """
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
        print(f"\nFetching data for {pair}...")

        # Fetch historical Klines
        data = get_historical_klines(
            symbol=pair,
            interval=args.interval,
            start_time=start_time,
            limit=args.limit
        )

        if data:
            print(f"Retrieved {len(data)} candles for {pair} ({args.interval}) over the past {args.days} days.")
            print("Sample candle:", data[0])

            output = {
                "symbol": pair,
                "interval": args.interval,
                "days": args.days,
                "limit": args.limit,
                "data": data
            }

            # Generate a descriptive filename
            filename = f"{pair}_{args.interval}_{args.days}days_{args.limit}candles.json"
            file_path = os.path.join(DATA_DIR, filename)

            # Save the data to JSON
            with open(file_path, "w") as json_file:
                json.dump(output, json_file, indent=4)

            print(f"\nData saved to {file_path}")

            try:
                collection.delete_many({"symbol": pair})
                collection.insert_one(output)
                print(f"\nData successfully committed to MongoDB")
            except Exception as e:
                print(f"MongoDB error for {pair}: {e}")