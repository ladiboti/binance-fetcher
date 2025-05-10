import os
import subprocess
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
POSTGRES_URI = os.getenv("POSTGRES_URI")
MIGRATIONS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../database"))

# Parse the PostgreSQL URI
parsed_uri = urlparse(POSTGRES_URI)
host = parsed_uri.hostname
port = parsed_uri.port or 5432
user = parsed_uri.username
password = parsed_uri.password
database = parsed_uri.path[1:]  # Remove leading "/"

PSQL_PATH = "/usr/sbin/psql" 

def run_migration(filename):
    migration_file = os.path.join(MIGRATIONS_DIR, filename)
    logger.info(f"Running migration: {filename}")
    
    # Set PGPASSWORD environment variable for passwordless authentication
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    try:
        result = subprocess.run(
            [PSQL_PATH, "-h", host, "-p", str(port), "-U", user, "-d", database, "-f", migration_file],
            capture_output=True,
            text=True,
            check=True,
            env=env  
        )
        logger.info(f"Successfully ran migration: {filename}")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running migration {filename}: {e}")
        logger.error(e.stderr)

def run_migration_pipeline():
    logger.info(f"Running migrations from directory: {MIGRATIONS_DIR}")
    
    migrations = [
        "delete-tables.psql",
        "create-currencies.psql",
        "create-historical-prices.psql",
        "create-cluster-changes.psql",
        "create-snapshots.psql"
    ]
    
    for migration in migrations:
        run_migration(migration)

if __name__ == "__main__":
    run_migration_pipeline()