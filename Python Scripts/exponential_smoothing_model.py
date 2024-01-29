from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate_model(data, trend='add', seasonal=None):
    """
    Train and evaluate an Exponential Smoothing model on the time series data.
    
    Parameters:
    - data (array-like): Time series data to train the model on.
    - trend (str): Type of trend component: 'add' for additive, 'mul' for multiplicative, or None for no trend.
    - seasonal (str): Type of seasonal component: 'add' for additive, 'mul' for multiplicative, or None for no seasonal component.
    
    Returns:
    - float: Root Mean Squared Error (RMSE) of the model.
    """
    try:
        # Convert data to a numpy array
        data_array = np.array(data)
        
        # Train the Exponential Smoothing model
        model = ExponentialSmoothing(data_array, trend=trend, seasonal=seasonal)
        fitted_model = model.fit()
        
        # Make predictions
        predictions = fitted_model.predict(start=0, end=len(data)-1)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(data, predictions))
        
        # Visualize actual vs. predicted values
        visualize_predictions(data, predictions)
        
        return rmse
    except Exception as e:
        print("An error occurred during model training and evaluation:", e)
        return None

def visualize_predictions(actual, predicted):
    """
    Visualize actual vs. predicted values.
    
    Parameters:
    - actual (array-like): Actual time series data.
    - predicted (array-like): Predicted values from the model.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Exponential Smoothing Model - Actual vs. Predicted')
        plt.legend()
        plt.show()
    except Exception as e:
        print("An error occurred during visualization:", e)
