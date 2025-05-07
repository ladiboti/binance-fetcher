import os
import json
import argparse
import pymongo
import logging
import shutil
from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
from src.services import get_historical_klines
from src.models import run_clustering_pipeline

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
CHARTS_DIR = "charts"
MONGO_URI = os.getenv("MONGO_URI")

# Initialize MongoDB connection
client = pymongo.MongoClient(MONGO_URI)
db = client["crypto-db"] 
collection = db["klines"]  

# List of trading pairs to fetch
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

# Helper functions
def clear_directory(directory):
    """Clear all files in a directory"""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(directory)


def fetch_and_save_data(interval, days, limit):
    """Fetch data for all trading pairs and save to files and database"""
    results = []
    
    # Compute the start time based on the number of days
    start_time = datetime.now() - timedelta(days=days)
    
    for pair in trading_pairs:
        logger.info(f"Fetching data for {pair}...")
        
        try:
            # Fetch historical Klines
            data = get_historical_klines(
                symbol=pair,
                interval=interval,
                start_time=start_time,
                limit=limit
            )
            
            # Check if trading pair is listed
            if not data or len(data[0]) < 14:
                logger.warning(f"Pair not listed on Binance or no data available for {pair}")
                results.append({"symbol": pair, "status": "error", "message": "No data available"})
                continue
            
            logger.info(f"Retrieved {len(data)} candles for {pair} ({interval}) over the past {days} days.")
            
            # Prepare JSON output
            output = {
                "symbol": pair,
                "interval": interval,
                "days": days,
                "limit": limit,
                "data": data
            }
            
            # Generate a descriptive filename
            filename = f"{pair}_{interval}.json"
            file_path = os.path.join(DATA_DIR, filename)
            
            # Save the data to JSON
            try:
                with open(file_path, "w") as json_file:
                    json.dump(output, json_file, indent=4)
                logger.info(f"Data saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save {file_path}: {e}")
                results.append({"symbol": pair, "status": "error", "message": f"Failed to save file: {str(e)}"})
                continue
            
            # Save to MongoDB
            try:
                collection.delete_many({"symbol": pair})
                collection.insert_one(output)
                logger.info(f"Data successfully committed to MongoDB for {pair}")
                results.append({"symbol": pair, "status": "success", "candles": len(data)})
            except Exception as e:
                logger.error(f"MongoDB error for {pair}: {e}")
                results.append({"symbol": pair, "status": "error", "message": f"MongoDB error: {str(e)}"})
            
        except IndexError as e:
            logger.error(f"Pair not listed on Binance or invalid response for {pair}: {e}")
            results.append({"symbol": pair, "status": "error", "message": f"Invalid response: {str(e)}"})
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {pair}: {e}")
            results.append({"symbol": pair, "status": "error", "message": f"Unexpected error: {str(e)}"})
    
    return results


# API Routes
@app.route('/api/fetch', methods=['POST'])
def fetch_data():
    """API endpoint to fetch historical data"""
    try:
        # Get parameters from request
        data = request.get_json() or {}
        interval = data.get('interval', Client.KLINE_INTERVAL_1HOUR)
        days = int(data.get('days', 30))
        limit = int(data.get('limit', 1000))
        
        # Clear data directory
        clear_directory(DATA_DIR)
        
        # Fetch and save data
        results = fetch_and_save_data(interval, days, limit)
        
        return jsonify({
            "status": "success",
            "message": f"Data fetched for {len(results)} trading pairs",
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in fetch_data endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/cluster', methods=['POST'])
def cluster_data():
    """API endpoint to run clustering on the fetched data"""
    try:
        # Clear charts directory
        clear_directory(CHARTS_DIR)
        
        # Run clustering pipeline
        result = run_clustering_pipeline()
        
        return jsonify({
            "status": "success",
            "message": "Clustering completed successfully",
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error in cluster_data endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/charts/<path:filename>')
def get_chart(filename):
    """API endpoint to retrieve generated charts"""
    return send_from_directory(CHARTS_DIR, filename)


@app.route('/api/data/<path:filename>')
def get_data(filename):
    """API endpoint to retrieve data files"""
    return send_from_directory(DATA_DIR, filename)


@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to check server status"""
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "data_files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else [],
        "chart_files": os.listdir(CHARTS_DIR) if os.path.exists(CHARTS_DIR) else []
    })

# Command line interface for direct execution
def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Binance Klines and run clustering.")
    parser.add_argument("--interval", type=str, default=Client.KLINE_INTERVAL_1HOUR, help="Candle interval (e.g., 1m, 5m, 1h, 1d).")
    parser.add_argument("--days", type=int, default=30, help="Number of past days to fetch.")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of candles per request (max: 1000).")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask server on.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch data without running the server.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    if args.fetch_only:
        # Just fetch data and exit
        clear_directory(DATA_DIR)
        fetch_and_save_data(args.interval, args.days, args.limit)
        run_clustering_pipeline()
    else:
        # Run as a web server
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
