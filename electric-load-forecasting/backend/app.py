from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app)

# Define model paths
RF_MODEL_PATH = 'random_forest_model.pkl'
XGB_MODEL_PATH = 'xgboost_model.pkl'
DATA_PATH = 'merged_weather_demand_final.csv'

# Load models
rf_model = joblib.load(RF_MODEL_PATH)
try:
    xgb_model = joblib.load(XGB_MODEL_PATH)
    print(f"Successfully loaded XGBoost model from {XGB_MODEL_PATH}")
except Exception as e:
    print(f"Error loading XGBoost model: {str(e)}")
    xgb_model = None

# Load data
data = pd.read_csv(DATA_PATH)

data['DateTime'] = pd.to_datetime(data['DateTime'])
cities = data['City'].unique().tolist()

# Define available models
available_models = ['random_forest']
if xgb_model is not None:
    available_models.append('xgboost')

# Load or initialize encoder for categorical features
ENCODER_PATH = 'categorical_encoder.pkl'
if os.path.exists(ENCODER_PATH):
    encoder = joblib.load(ENCODER_PATH)
    print(f"Loaded encoder from {ENCODER_PATH}")
else:
    # Initialize encoder with all possible categorical values
    # This is a fallback and should be replaced by loading the actual encoder used during training
    categorical_features = ['Weekday', 'Month', 'Season', 'Sub-Region']
    
    # Handle different scikit-learn versions - try both parameter names
    try:
        # For newer scikit-learn versions (>=1.2)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        # For older scikit-learn versions
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    encoder.fit(data[categorical_features].fillna('unknown'))
    print(f"Initialized new encoder for {categorical_features}")
    joblib.dump(encoder, ENCODER_PATH)

@app.route('/api/cities', methods=['GET'])
def get_cities():
    return jsonify(cities)

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(available_models)

@app.route('/api/date_range', methods=['GET'])
def get_date_range():
    min_date = data['DateTime'].min().strftime('%Y-%m-%d')
    max_date = data['DateTime'].max().strftime('%Y-%m-%d')
    return jsonify({'min_date': min_date, 'max_date': max_date})

def apply_feature_engineering(city_data):
    """Apply the same feature engineering steps used during model training"""
    print("Starting feature engineering...")
    print(f"Initial data shape: {city_data.shape}")
    
    # Make a copy and ensure it's sorted by datetime
    df_features = city_data.copy()
    df_features = df_features.sort_values('DateTime')
    
    # Check if we have sufficient data for lag features
    print(f"Date range in data: {df_features['DateTime'].min()} to {df_features['DateTime'].max()}")
    time_span = df_features['DateTime'].max() - df_features['DateTime'].min()
    print(f"Time span in data: {time_span}")
    
    if len(df_features) < 24*8:  # At least 8 days of hourly data
        print(f"WARNING: Dataset may be too small ({len(df_features)} rows) for reliable lag features")
        
    # 1. Create lag features (previous days demand)
    print("Creating lag features...")
    
    # Add lag features (previous 24 hours demand)
    df_features['demand_lag_24h'] = df_features.groupby('City')['Demand (MW)'].shift(24)
    
    # Add lag features for previous day same hour (24, 48, 72 hours)
    for i in range(1, 4):
        df_features[f'demand_lag_{i}d_same_hour'] = df_features.groupby('City')['Demand (MW)'].shift(i*24)
    
    # 2. Add moving averages for smoothed signals
    print("Creating moving averages...")
    df_features['demand_ma_24h'] = df_features.groupby('City')['Demand (MW)'].rolling(window=24).mean().reset_index(level=0, drop=True)
    df_features['demand_ma_7d'] = df_features.groupby('City')['Demand (MW)'].rolling(window=24*7).mean().reset_index(level=0, drop=True)
    
    # 3. Add daily min, max, and range
    print("Creating daily stats...")
    df_features['day_of_year'] = df_features['DateTime'].dt.dayofyear
    
    # Group by city and day of year for daily statistics
    daily_stats = df_features.groupby(['City', 'day_of_year'])['Demand (MW)'].agg(['min', 'max'])
    daily_stats['range'] = daily_stats['max'] - daily_stats['min']
    
    # Reset index to prepare for merge
    daily_stats = daily_stats.reset_index()
    
    # Merge with original dataframe
    df_features = df_features.merge(daily_stats, on=['City', 'day_of_year'], how='left')
    df_features.rename(columns={'min': 'daily_min', 'max': 'daily_max', 'range': 'daily_range'}, inplace=True)
    
    # 4. Add Cyclical Features
    print("Creating cyclical features...")
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['Hour']/24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['Hour']/24)
    df_features['weekday_sin'] = np.sin(2 * np.pi * df_features['Weekday']/7)
    df_features['weekday_cos'] = np.cos(2 * np.pi * df_features['Weekday']/7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['Month']/12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['Month']/12)
    
    # 5. Weather derivatives
    print("Creating weather derivatives...")
    df_features['temp_change_24h'] = df_features.groupby('City')['Temperature (F)'].shift(24)
    df_features['temp_change_24h'] = df_features['Temperature (F)'] - df_features['temp_change_24h']
    
    # 6. Create peak/off-peak indicator (6am-10pm is peak)
    df_features['is_peak_hours'] = ((df_features['Hour'] >= 6) & (df_features['Hour'] <= 22)).astype(int)
    
    print(f"Feature engineering complete. Shape before handling NaN: {df_features.shape}")
    print(f"NaN counts: {df_features[['demand_lag_24h', 'demand_ma_24h', 'demand_ma_7d']].isna().sum()}")
    
    return df_features

def prepare_features_for_prediction(df_features, start_date, end_date):
    """Prepare features for model prediction by selecting rows and handling missing values"""
    # 1. Define feature lists
    categorical_features = ['Weekday', 'Month', 'Season', 'Sub-Region']
    numerical_features = [
        'Temperature (F)', 'Humidity', 'Wind Speed (mph)', 'Pressure', 'UV Index',
        'Cloud Cover', 'Hour', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos', 'is_peak_hours', 'demand_lag_24h',
        'demand_lag_1d_same_hour', 'demand_lag_2d_same_hour', 'demand_lag_3d_same_hour',
        'temp_change_24h', 'demand_ma_24h', 'demand_ma_7d'
    ]
    
    # 2. Select only the rows in the target forecast window
    prediction_df = df_features[(df_features['DateTime'] >= start_date) & 
                               (df_features['DateTime'] <= end_date)].copy()
    
    print(f"Prediction window shape: {prediction_df.shape}")
    
    # 3. Handle missing values
    # For numerical features, use median from the feature-engineered dataset
    for col in numerical_features:
        if col in prediction_df.columns and prediction_df[col].isnull().any():
            # First try to use median from the current dataset
            median_val = df_features[col].median()
            # If that's NaN too, use 0
            if pd.isna(median_val):
                median_val = 0
            prediction_df[col] = prediction_df[col].fillna(median_val)
    
    # For categorical features, use mode
    for col in categorical_features:
        if col in prediction_df.columns and prediction_df[col].isnull().any():
            # Use mode from the current dataset
            mode_val = df_features[col].mode()[0]
            prediction_df[col] = prediction_df[col].fillna(mode_val)
    
    # 4. Ensure all required features exist
    for feature in numerical_features:
        if feature not in prediction_df.columns:
            print(f"WARNING: Feature {feature} is missing and will be created with zeros")
            prediction_df[feature] = 0
    
    # Handle categorical features using one-hot encoding
    # First check if all categories are present
    for feature in categorical_features:
        if feature not in prediction_df.columns:
            print(f"WARNING: Categorical feature {feature} is missing and will be created with default value")
            prediction_df[feature] = 'unknown'
    
    # Transform categorical features using the encoder
    print("Transforming categorical features...")
    X_categorical = encoder.transform(prediction_df[categorical_features].fillna('unknown'))
    
    # Get encoded feature names
    encoded_feature_names = []
    for idx, feature in enumerate(categorical_features):
        categories = encoder.categories_[idx]
        for category in categories:
            encoded_feature_names.append(f"{feature}_{category}")
    
    # Create DataFrame with encoded features
    cat_df = pd.DataFrame(X_categorical, columns=encoded_feature_names, index=prediction_df.index)
    
    # Extract numerical features
    num_df = prediction_df[numerical_features]
    
    # Combine numerical and categorical features
    X = pd.concat([num_df, cat_df], axis=1)
    
    # Final check for NaNs
    if X.isna().any().any():
        print("WARNING: NaN values still present in features. Filling with 0.")
        X = X.fillna(0)
    
    print(f"Final feature matrix shape: {X.shape}")
    
    return prediction_df, X

@app.route('/api/forecast', methods=['POST'])
def forecast():
    request_data = request.json
    city = request_data.get('city', 'la')
    start_date = pd.to_datetime(request_data.get('start_date'))
    end_date = pd.to_datetime(request_data.get('end_date'))
    model_type = request_data.get('model_type', 'random_forest')
    
    # Check if requested model is available
    if model_type not in available_models:
        return jsonify({'error': f'Model {model_type} is not available'})
    
    print(f"\n--- FORECAST REQUEST ---")
    print(f"City: {city}, Date range: {start_date} to {end_date}, Model: {model_type}")
    
    # Select the appropriate model
    if model_type == 'random_forest':
        model = rf_model
    elif model_type == 'xgboost':
        model = xgb_model
    else:
        return jsonify({'error': f'Unknown model type: {model_type}'})
    
    # Calculate how much history we need for feature engineering
    # Need at least 7 days for weekly moving average
    history_start = start_date - timedelta(days=30)  # Generous buffer for feature engineering
    
    print(f"Fetching data from {history_start} to {end_date}")
    
    # Load data
    raw_data = data[(data['City'] == city) & 
                   (data['DateTime'] >= history_start) & 
                   (data['DateTime'] <= end_date)].copy()
    
    if raw_data.empty:
        return jsonify({'error': 'No data available for the selected city and date range'})
    
    print(f"Retrieved {len(raw_data)} rows of raw data")
    
    try:
        # Step 1: Apply feature engineering to the entire dataset (including history)
        features_df = apply_feature_engineering(raw_data)
        
        # Step 2: Prepare features for the prediction window and handle missing values
        prediction_df, X = prepare_features_for_prediction(features_df, start_date, end_date)
        
        if prediction_df.empty:
            return jsonify({'error': 'No data available for prediction after feature engineering'})
        
    except Exception as e:
        print(f"Feature engineering error: {str(e)}")
        return jsonify({'error': f'Feature engineering failed: {str(e)}'})
    
    try:
        # Make predictions
        predictions = model.predict(X)
        print(f"Successfully generated {len(predictions)} predictions with {model_type} model")
        
        # Build response
        result = []
        for idx, (i, row) in enumerate(prediction_df.iterrows()):
            if idx < len(predictions):  # Ensure we don't go out of bounds
                result.append({
                    'date': row['DateTime'].strftime('%Y-%m-%d %H:%M'),
                    'actual': float(row['Demand (MW)']),
                    'predicted': float(predictions[idx]),
                    'model': model_type
                })
            else:
                print(f"WARNING: Index {idx} is out of bounds for predictions array of size {len(predictions)}")
                result.append({
                    'date': row['DateTime'].strftime('%Y-%m-%d %H:%M'),
                    'actual': float(row['Demand (MW)']),
                    'predicted': None,  # No prediction available
                    'model': model_type
                })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Model prediction failed: {str(e)}'})


@app.route('/api/clustering', methods=['POST'])
def clustering():
    request_data = request.json
    city = request_data.get('city', 'la')
    k = request_data.get('k', 3)

    # Filter by city and copy
    city_data = data[data['City'] == city].copy()
    if city_data.empty:
        return jsonify({'error': 'No data found for the selected city'})

    # Features to use for clustering
    cluster_features = ['Demand (MW)', 'Hour', 'Weekend']

    # Ensure all features are numeric
    for feature in cluster_features:
        city_data[feature] = pd.to_numeric(city_data[feature], errors='coerce')

    # Drop NaNs
    city_data = city_data.dropna(subset=cluster_features)

    # Optional: remove near-constant rows
    city_data = city_data[city_data[cluster_features].std(axis=1) > 1e-2]

    # Now that city_data is cleaned, generate X from it
    X = city_data[cluster_features].values

    # Check if data is still valid
    if len(X) < k:
        return jsonify({'error': f'Not enough data points for {k} clusters'})

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Prepare results
    result = []
    for i in range(len(X_pca)):
        result.append({
            'x': float(X_pca[i, 0]),
            'y': float(X_pca[i, 1]),
            'cluster': int(clusters[i]),
            'demand': float(city_data.iloc[i]['Demand (MW)']),
            'temperature': float(city_data.iloc[i].get('Temperature (F)', 0)),
            'date': city_data.iloc[i]['DateTime'].strftime('%Y-%m-%d %H:%M')
        })

    print(f"[CLUSTERING DEBUG] Cleaned rows: {len(city_data)} | Returned points: {len(result)}")
    return jsonify(result)

#if __name__ == '__main__':
    #app.run(debug=True, port=5000)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
