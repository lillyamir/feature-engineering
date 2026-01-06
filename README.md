# Avocado Price Prediction with Ridge and Lasso Regression

A machine learning project demonstrating feature engineering, hyperparameter tuning, and model comparison using Ridge and Lasso regression to predict average avocado prices.

## Project Overview

This project uses a Kaggle dataset of avocado prices across different U.S. regions to build and compare regularized linear regression models. The focus is on proper feature engineering techniques, handling categorical variables, and model evaluation.

## Key Techniques Demonstrated

### Feature Engineering
- **Target Encoding**: Efficiently encoded 44 city-level regions using mean price values (avoiding 44 one-hot encoded columns)
- **Date Features**: Extracted month, quarter, and day of year from date information to capture seasonal patterns
- **One-Hot Encoding**: Converted avocado type (conventional vs. organic) to binary features
- **Feature Scaling**: Standardized all features using StandardScaler for optimal regularization performance

### Model Development
- **Train-Test Split**: 80-20 split with fixed random state for reproducibility
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation to find optimal regularization strength (alpha/λ)
- **Model Comparison**: Evaluated both Ridge and Lasso regression using multiple metrics (MSE, RMSE, MAE, R²)

### Data Quality
- Filtered to city-level regions only (removed aggregate regions like "TotalUS", "West", etc.)
- Cleaned data to ensure bag totals matched component values
- Proper handling of train-test leakage in target encoding

## Technologies Used
- Python 3.9+
- pandas
- scikit-learn
- numpy

## Model Performance

The models were evaluated on:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

Ridge and Lasso were compared to select the best performing model based on test set R² scores.

## Key Learnings

This project demonstrates:
1. How to handle categorical variables efficiently using target encoding
2. Proper implementation of target encoding to avoid data leakage
3. The difference between Ridge (L2) and Lasso (L1) regularization in practice
4. Feature engineering for time-series data
5. Cross-validation and hyperparameter tuning best practices

## Dataset

Avocado prices dataset from Kaggle, containing historical data on avocado prices, volumes, and regions across the United States from 2015-2018.
