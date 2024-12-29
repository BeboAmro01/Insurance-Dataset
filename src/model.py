import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from preprocessing import load_and_preprocess_data, save_preprocessing_artifacts

def train_house_price_model(dataset_path, model_path, feature_names_path):
    # Load and preprocess data
    x_train, x_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    
    
    
    # Train Random Forest Regressor
    model1 = RandomForestRegressor(
    n_estimators=100, 
    random_state=42, 
    max_depth=10
    )
    
    model1.fit(x_train, y_train)    
    # Make predictions
    y_pred = model1.predict(x_test)    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model1, model_path)
    
    
    
    
    # Return performance metrics
    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R-squared': r2
    }

# If script is run directly, train the model
if __name__ == '__main__':
    metrics = train_house_price_model(
        dataset_path='../dataset/insurance.csv',
        model_path='trained_model.pkl',
        feature_names_path='feature_names.txt'
    )
    print("Model Training Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")