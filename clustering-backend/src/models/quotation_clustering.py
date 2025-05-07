import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import pymongo

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger


def setup_environment():
    """Set up the environment, directories and load environment variables"""
    logger = setup_logging()
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
    
    # Suppress FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Create directories
    EXPORT_DIR = "data"
    CHARTS_DIR = "charts"
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    return logger, EXPORT_DIR, CHARTS_DIR


def clean_data_directory(export_dir):
    """Delete all files in the data directory"""
    logger = logging.getLogger(__name__)
    logger.info(f"Deleting {export_dir}...")
    for filename in os.listdir(export_dir):
        file_path = os.path.join(export_dir, filename)
        try:
            os.remove(file_path)
            logger.info(f"Deleted:: {file_path}")
        except Exception as e:
            logger.error(f"Unable to delete: {file_path}, error: {e}")


def connect_to_mongodb():
    """Connect to MongoDB and return client, db, collection"""
    logger = logging.getLogger(__name__)
    logger.info("Connecting to MongoDB...")
    MONGO_URI = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(MONGO_URI)
    db = client["crypto-db"]
    collection = db["klines"]
    return client, db, collection


def download_documents(collection, export_dir):
    """Download documents from MongoDB and save them one-by-one"""
    logger = logging.getLogger(__name__)
    logger.info("Downloading documents from MongoDB...")
    documents = collection.find()
    for doc in documents:
        doc.pop("_id", None)  # Remove MongoDB internal ID
        filename = f"{doc['symbol']}_{doc['interval']}.json"
        file_path = os.path.join(export_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=4)
        logger.info(f"Saved: {file_path}")
    
    logger.info("All documents saved individually as JSON files.")


def load_json_files(export_dir):
    """Load all local JSON files into a combined DataFrame"""
    logger = logging.getLogger(__name__)
    logger.info("Loading JSON files into DataFrame...")
    json_files = glob.glob(os.path.join(export_dir, "*.json"))
    all_data = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            symbol = data.get('symbol', 'unknown')
            if "data" in data:
                for record in data["data"]:
                    record["symbol"] = symbol
                    all_data.append(record)
    
    combined_df = pd.DataFrame(all_data)
    logger.info(f"All local JSONs combined into a DataFrame! Total records loaded: {len(combined_df)}")
    return combined_df


def normalize_data(combined_df, export_dir):
    """Normalize data for each symbol and column group"""
    logger = logging.getLogger(__name__)
    logger.info("Normalizing data...")
    df_normalized = combined_df.copy()
    
    ohlc_columns = ['open', 'high', 'low', 'close']
    volume_columns = ['volume', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume']
    order_book_columns = [
        'best_bid_volume', 'best_ask_volume', 'bid_volume_depth', 'ask_volume_depth',
        'best_bid_price', 'best_ask_price', 'bid_ask_spread', 'order_book_imbalance'
    ]
    indicator_columns = [
        'sma_10', 'sma_50', 'ema_10', 'ema_50',
        'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
        'macd', 'macd_signal', 'atr', 'obv', 'rsi_14'
    ]
    
    # Iterate through unique symbols in the DataFrame
    for symbol in df_normalized['symbol'].unique():
        # Filter data for the current symbol
        symbol_data = df_normalized[df_normalized['symbol'] == symbol]
        
        # Create a new MinMaxScaler and StandardScaler instance
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()
        
        df_normalized.loc[symbol_data.index, ohlc_columns] = minmax_scaler.fit_transform(symbol_data[ohlc_columns])
        df_normalized.loc[symbol_data.index, volume_columns] = minmax_scaler.fit_transform(symbol_data[volume_columns])
        df_normalized.loc[symbol_data.index, order_book_columns] = standard_scaler.fit_transform(symbol_data[order_book_columns])
        df_normalized.loc[symbol_data.index, indicator_columns] = minmax_scaler.fit_transform(symbol_data[indicator_columns])
    
    # Save the normalized DataFrame to a CSV file in the data directory
    output_path = os.path.join(export_dir, 'df_normalized.csv')
    df_normalized.to_csv(output_path, index=False)
    logger.info(f'Normalized DataFrame saved to {output_path}')
    
    return df_normalized


def create_close_price_chart(df_normalized, charts_dir):
    """Create close price over time chart for each symbol"""
    logger = logging.getLogger(__name__)
    logger.info("Creating close price chart...")
    df_normalized['timestamp'] = pd.to_datetime(df_normalized['timestamp'], unit='ms')
    plt.figure(figsize=(15, 6))
    for symbol in df_normalized['symbol'].unique():
        symbol_data = df_normalized[df_normalized['symbol'] == symbol]
        plt.plot(symbol_data['timestamp'], symbol_data['close'], label=symbol)
    plt.title('Close Price Over Time by Symbol')
    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    plt.legend(title='Symbol')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'close_price_over_time.png'))
    plt.close()
    logger.info(f"Close price chart saved to {os.path.join(charts_dir, 'close_price_over_time.png')}")


def calculate_volatility_correlation(df_normalized, charts_dir):
    """Calculate volatility and correlation, visualize correlation matrix"""
    logger = logging.getLogger(__name__)
    logger.info("Calculating volatility and correlation...")
    volatility_df = df_normalized.copy()
    # volatility_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last', inplace=True)
    volatility_df['price_change'] = volatility_df.groupby('symbol')['close'].pct_change()
    volatility_pivot = volatility_df.pivot(index='timestamp', columns='symbol', values='price_change')
    correlation_matrix = volatility_pivot.corr()
    logger.info("Correlation matrix calculated")
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Correlation Matrix of Price Volatility')
    plt.savefig(os.path.join(charts_dir, 'correlation_matrix.png'))
    plt.close()
    logger.info(f"Correlation matrix saved to {os.path.join(charts_dir, 'correlation_matrix.png')}")


def perform_feature_selection(df_normalized):
    """Run feature selection using RFE"""
    logger = logging.getLogger(__name__)
    logger.info("Running feature selection...")
    X = df_normalized.drop(columns=['high', 'timestamp', 'close_time']).select_dtypes(include=['number'])
    y = df_normalized['high']
    model = Ridge(alpha=1.0, solver='saga', random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    n_features_to_select = 10
    
    logger.info(f"Running TimeSeries RFE on {X.shape[1]} features...")
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    X_selected = X[selected_features]
    scores = cross_val_score(model, X_selected, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-scores.mean())
    logger.info(f"TimeSeries RFE completed. Selected {n_features_to_select} features with average RMSE: {rmse:.4f}")
    logger.info(f"Selected Features: {list(selected_features)}")
    
    # Create DataFrame with selected features
    df_linear_regression_features = df_normalized[['timestamp', 'close_time', 'symbol']].join(X_selected)
    logger.info("Created DataFrame with selected features")
    
    return df_linear_regression_features


def run_pca_analysis(df_linear_regression_features, charts_dir):
    """Run PCA analysis and create visualizations"""
    logger = logging.getLogger(__name__)
    logger.info("Running PCA analysis...")
    X = df_linear_regression_features.drop(columns=['timestamp', 'close_time', 'symbol'])
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_result, columns=['Főkomponens 1', 'Főkomponens 2', 'Főkomponens 3'])
    pca_df['symbol'] = df_linear_regression_features['symbol'].values
    logger.info("PCA analysis completed")
    
    # Save 3D PCA visualization
    fig = px.scatter_3d(
        pca_df,
        x='Főkomponens 1',
        y='Főkomponens 2',
        z='Főkomponens 3',
        color='symbol',
        hover_data=['symbol'],
        title='PCA 3D Vizualizáció Symbol-ok szerint',
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        scene=dict(
            xaxis=dict(title='Főkomponens 1', backgroundcolor='white', color='black', gridcolor='lightgrey'),
            yaxis=dict(title='Főkomponens 2', backgroundcolor='white', color='black', gridcolor='lightgrey'),
            zaxis=dict(title='Főkomponens 3', backgroundcolor='white', color='black', gridcolor='lightgrey')
        )
    )
    fig.write_html(os.path.join(charts_dir, 'pca_3d.html'))
    logger.info(f"3D PCA visualization saved to {os.path.join(charts_dir, 'pca_3d.html')}")


def create_pca_time_series(df_linear_regression_features, charts_dir):
    """Create PCA time series visualizations"""
    logger = logging.getLogger(__name__)
    logger.info("Creating PCA time series visualizations...")
    X_selected = df_linear_regression_features.drop(columns=['timestamp', 'close_time', 'symbol'])
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_selected)
    pca_df = pd.DataFrame(pca_result, columns=['Főkomponens 1', 'Főkomponens 2', 'Főkomponens 3'])
    pca_df['timestamp'] = df_linear_regression_features['timestamp']
    pca_df['symbol'] = df_linear_regression_features['symbol']
    pca_df['timestamp'] = pd.to_datetime(pca_df['timestamp'], unit='ms')
    
    pca_axis_mapping = {
        'Főkomponens 1': 'Combination of Open, Close and SMAs',
        'Főkomponens 2': 'Volume-related features',
        'Főkomponens 3': 'Bollinger Bands and EMA patterns'
    }
    
    for component in ['Főkomponens 1', 'Főkomponens 2', 'Főkomponens 3']:
        fig = px.line(
            pca_df,
            x='timestamp',
            y=component,
            color='symbol',
            title=f"Time Series Analysis of {component}",
            labels={
                'timestamp': 'Timestamp',
                component: 'Principal Component Value',
                'symbol': 'Symbol'
            },
            template='plotly_white'
        )
        
        dimension_description = pca_axis_mapping.get(component, '')
        fig.add_annotation(
            text=f"Axis Composition: {dimension_description}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15, showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        fig.update_layout(
            xaxis_title='Timestamp',
            yaxis_title='Principal Component Value',
            legend_title='Symbol',
            legend=dict(
                title_font=dict(size=12),
                font=dict(size=10),
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1
            )
        )
        
        fig.write_html(os.path.join(charts_dir, f'pca_timeseries_{component}.html'))
        logger.info(f"PCA time series for {component} saved to {os.path.join(charts_dir, f'pca_timeseries_{component}.html')}")


def perform_clustering(df_linear_regression_features, charts_dir):
    """Run agglomerative clustering and create visualizations"""
    logger = logging.getLogger(__name__)
    logger.info("Running agglomerative clustering...")
    X_cluster = df_linear_regression_features.drop(columns=['timestamp', 'close_time', 'symbol'])
    agg_cluster = AgglomerativeClustering(n_clusters=6, linkage='ward')
    df_linear_regression_features['agg_cluster'] = agg_cluster.fit_predict(X_cluster)
    logger.info("Clustering completed")
    
    # PCA with clustering
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_cluster)
    pca_df = pd.DataFrame(pca_result, columns=['Főkomponens 1', 'Főkomponens 2', 'Főkomponens 3'])
    pca_df['symbol'] = df_linear_regression_features['symbol'].values
    pca_df['agg_cluster'] = df_linear_regression_features['agg_cluster'].values
    
    # 3D cluster visualization
    fig_3d = px.scatter_3d(
        pca_df,
        x='Főkomponens 1',
        y='Főkomponens 2',
        z='Főkomponens 3',
        color='agg_cluster',
        symbol='symbol',
        labels={'agg_cluster': 'Klaszter', 'symbol': 'Szimbólum'},
        title='Agglomeratív klaszterek 3D-ben',
        template='plotly_white',
        opacity=0.8,
        hover_data=['symbol']
    )
    fig_3d.update_traces(marker=dict(size=5))
    fig_3d.write_html(os.path.join(charts_dir, 'cluster_3d.html'))
    logger.info(f"3D cluster visualization saved to {os.path.join(charts_dir, 'cluster_3d.html')}")
    
    # Cluster time series visualizations
    pca_df['timestamp'] = df_linear_regression_features['timestamp'].values
    for component in ['Főkomponens 1', 'Főkomponens 2', 'Főkomponens 3']:
        fig_time_series = px.scatter(
            pca_df,
            x='timestamp',
            y=component,
            color='agg_cluster',
            symbol='symbol',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'agg_cluster': 'Klaszter', 'symbol': 'Szimbólum'},
            title=f"Agglomeratív klaszterek idősorosan ({component})",
            template="plotly_white",
            opacity=0.8
        )
        
        fig_time_series.update_traces(marker=dict(size=5))
        fig_time_series.update_layout(
            legend=dict(
                title="Clusters",
                x=0.99, xanchor="right",
                y=0.99, yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.6)"
            )
        )
        
        fig_time_series.write_html(os.path.join(charts_dir, f'cluster_timeseries_{component}.html'))
        logger.info(f"Cluster time series for {component} saved to {os.path.join(charts_dir, f'cluster_timeseries_{component}.html')}")
    
    return df_linear_regression_features


def export_cluster_assignments(df_linear_regression_features, export_dir, client):
    """Export cluster assignments to CSV and MongoDB"""
    logger = logging.getLogger(__name__)
    logger.info("Exporting cluster assignments...")
    
    cluster_assignments = df_linear_regression_features[['timestamp', 'symbol', 'agg_cluster']].copy()
    cluster_assignments.rename(columns={'agg_cluster': 'cluster_id'}, inplace=True)
    cluster_assignments['timestamp'] = pd.to_datetime(cluster_assignments['timestamp'], unit='ms')
    batch_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    cluster_assignments['batch_id'] = batch_id
    
    csv_path = os.path.join(export_dir, "cluster_assignments.csv")
    cluster_assignments.to_csv(csv_path, index=False)
    
    logger.info(f"Exported: {csv_path} (total {len(cluster_assignments)} rows)")
    
    # Commit csv to MongoDB
    logger.info("Committing cluster assignments to MongoDB...")
    
    cluster_collection = client["crypto-db"]["cluster_assignments"]
    
    # Delete duplicants by batch ID
    cluster_collection.delete_many({"batch_id": batch_id})
    
    mongo_docs = cluster_assignments.to_dict(orient="records")
    
    # Insert
    cluster_collection.insert_many(mongo_docs)
    logger.info(f"Committed {len(mongo_docs)} cluster assignments to MongoDB (batch_id={batch_id})")


def run_clustering_pipeline():
    """Main function to orchestrate the entire crypto analysis process"""
    # Setup
    logger, EXPORT_DIR, CHARTS_DIR = setup_environment()
    
    # Clean data directory
    clean_data_directory(EXPORT_DIR)
    
    # Connect to MongoDB
    client, db, collection = connect_to_mongodb()
    
    # Download documents
    download_documents(collection, EXPORT_DIR)
    
    # Load data
    combined_df = load_json_files(EXPORT_DIR)
    
    # Normalize data
    df_normalized = normalize_data(combined_df, EXPORT_DIR)
    
    # Create charts
    create_close_price_chart(df_normalized, CHARTS_DIR)
    
    # Calculate volatility and correlation
    calculate_volatility_correlation(df_normalized, CHARTS_DIR)
    
    # Perform feature selection
    df_linear_regression_features = perform_feature_selection(df_normalized)
    
    # Run PCA analysis
    run_pca_analysis(df_linear_regression_features, CHARTS_DIR)
    
    # Create PCA time series visualizations
    create_pca_time_series(df_linear_regression_features, CHARTS_DIR)
    
    # Perform clustering
    df_linear_regression_features = perform_clustering(df_linear_regression_features, CHARTS_DIR)
    
    # Export cluster assignments
    export_cluster_assignments(df_linear_regression_features, EXPORT_DIR, client)
    
    logger.info("Crypto analysis completed successfully.")
    return True

# Main execution point
if __name__ == "__main__":
    run_clustering_pipeline()