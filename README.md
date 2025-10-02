# Electric Load Forecasting & Demand Analysis
A comprehensive data mining project that forecasts electricity demand using historical load data, weather information, and machine learning models.

## Project Overview
This system analyzes and predicts electricity consumption patterns across major US cities by combining:

Historical demand data from EIA (Energy Information Administration)

Weather conditions from city-specific JSON files

Time-based features and seasonal patterns

Multiple machine learning models for accurate forecasting

## Dataset
Sources:

Electricity demand: 4 EIA CSV files (July 2018 - May 2020)

Weather data: 10 city JSON files (Dallas, Houston, LA, NYC, Philadelphia, Phoenix, San Antonio, San Diego, San Jose, Seattle)

Preprocessing:

Consolidated 165,192 records with 21 features

Handled missing values and anomalies

Engineered time-based features (hour, weekday, season, weekend indicators)

Normalized demand and weather features

## Clustering Analysis
Identified distinct electricity consumption patterns using:

K-Means (Best Performance - Silhouette Score: 0.4750)

6 optimal clusters revealing peak demand periods and time-of-day usage patterns

Clear separation between weekday vs weekend consumption

Hierarchical Clustering

4 clusters focusing on hourly behavior and night-time demand trends

DBSCAN

Density-based approach confirming weekday/weekend patterns

## Forecasting Models
Tested multiple algorithms with SARIMA emerging as the top performer:

Model	MAE (MW)	RMSE (MW)	MAPE (%)	R²
SARIMA	330.70	431.95	11.83%	0.9512
Random Forest	556.96	977.42	20.74%	0.9080
Neural Network	698.10	1063.18	24.95%	0.8911
XGBoost	738.89	1202.43	25.92%	0.8607
Linear Regression	859.96	1197.43	34.66%	0.8619


## Features
### Forecasting Capabilities
Multi-city electricity demand predictions

Support for Random Forest and XGBoost models

Feature engineering with lag variables, moving averages, and cyclical encoding

Actual vs predicted comparisons with detailed metrics

### Clustering Analysis
Interactive cluster visualization using PCA

Demand pattern discovery across different time periods

City-specific consumption behavior analysis

### Web Dashboard
React.js frontend with Recharts visualizations

Flask backend API with model serving

Interactive date range selection and city filtering

Real-time clustering parameter adjustment

## Technical Architecture
### Backend (Flask):

Model loading and inference

Feature engineering pipeline

Data preprocessing and validation

REST API endpoints for forecasting and clustering

### Frontend (React):

Interactive charts and visualizations

Responsive design for desktop and tablet

Real-time data filtering and exploration

User-friendly control panels

### Machine Learning:

Multiple model support with consistent interface

Comprehensive evaluation metrics (MAE, RMSE, MAPE, R²)

Automated hyperparameter tuning and validation


## Key Findings
Seasonal patterns significantly influence electricity demand

Weather conditions (temperature, humidity) strongly correlate with consumption

Weekday vs weekend usage shows distinct behavioral clusters

Time-of-day patterns reveal peak demand periods critical for load balancing
