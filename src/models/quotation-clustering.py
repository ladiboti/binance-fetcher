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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Create directories
EXPORT_DIR = "data"
CHARTS_DIR = "charts"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Delete data directory
logger.info(f"Deleting {EXPORT_DIR}...")
for filename in os.listdir(EXPORT_DIR):
    file_path = os.path.join(EXPORT_DIR, filename)
    try:
        os.remove(file_path)
        logger.info(f"Deleted:: {file_path}")
    except Exception as e:
        logger.error(f"Unable to delete: {file_path}, error: {e}")

# MongoDB connection
logger.info("Connecting to MongoDB...")
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["crypto-db"]
collection = db["klines"]

# Step 1: Download documents and save them one-by-one
logger.info("Downloading documents from MongoDB...")
documents = collection.find()
for doc in documents:
    doc.pop("_id", None)  # Remove MongoDB internal ID
    filename = f"{doc['symbol']}_{doc['interval']}.json"
    file_path = os.path.join(EXPORT_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=4)
    logger.info(f"Saved: {file_path}")

logger.info("All documents saved individually as JSON files.")

# Step 2: Load all local JSON files into a combined DataFrame
logger.info("Loading JSON files into DataFrame...")
json_files = glob.glob(os.path.join(EXPORT_DIR, "*.json"))
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

# Normalize data
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
output_path = os.path.join(EXPORT_DIR, 'df_normalized.csv')
df_normalized.to_csv(output_path, index=False)
logger.info(f'Normalized DataFrame saved to {output_path}')

# Plot close price over time
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
plt.savefig(os.path.join(CHARTS_DIR, 'close_price_over_time.png'))
plt.close()
logger.info(f"Close price chart saved to {os.path.join(CHARTS_DIR, 'close_price_over_time.png')}")

# Calculate volatility and correlation
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
plt.savefig(os.path.join(CHARTS_DIR, 'correlation_matrix.png'))
plt.close()
logger.info(f"Correlation matrix saved to {os.path.join(CHARTS_DIR, 'correlation_matrix.png')}")

# Feature selection
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

# PCA analysis
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
fig.write_html(os.path.join(CHARTS_DIR, 'pca_3d.html'))
logger.info(f"3D PCA visualization saved to {os.path.join(CHARTS_DIR, 'pca_3d.html')}")

# PCA with timestamp
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
    
    fig.write_html(os.path.join(CHARTS_DIR, f'pca_timeseries_{component}.html'))
    logger.info(f"PCA time series for {component} saved to {os.path.join(CHARTS_DIR, f'pca_timeseries_{component}.html')}")

# Clustering
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
fig_3d.write_html(os.path.join(CHARTS_DIR, 'cluster_3d.html'))
logger.info(f"3D cluster visualization saved to {os.path.join(CHARTS_DIR, 'cluster_3d.html')}")

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
    
    fig_time_series.write_html(os.path.join(CHARTS_DIR, f'cluster_timeseries_{component}.html'))
    logger.info(f"Cluster time series for {component} saved to {os.path.join(CHARTS_DIR, f'cluster_timeseries_{component}.html')}")

# Export cluster assignments
logger.info("Exporting cluster assignments...")

cluster_assignments = df_linear_regression_features[['timestamp', 'symbol', 'agg_cluster']].copy()
cluster_assignments.rename(columns={'agg_cluster': 'cluster_id'}, inplace=True)
cluster_assignments['timestamp'] = pd.to_datetime(cluster_assignments['timestamp'], unit='ms')
batch_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
cluster_assignments['batch_id'] = batch_id

csv_path = os.path.join(EXPORT_DIR, "cluster_assignments.csv")
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
