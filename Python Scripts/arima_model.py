from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

def train_arima_model(data, order=(1, 1, 1)):
    """
    Train an ARIMA model on the time series data.
    
    Parameters:
    - data (array-like): Time series data to train the model on.
    - order (tuple): ARIMA model order (p, d, q).
    
    Returns:
    - ARIMA: Trained ARIMA model.
    """
    # Convert data to a numpy array
    data_array = np.array(data)
    
    # Train the ARIMA model
    model = ARIMA(data_array, order=order)
    fitted_model = model.fit()
    
    return fitted_model

def evaluate_arima_model(model, test_data):
    """
    Evaluate an ARIMA model on test data.
    
    Parameters:
    - model (ARIMA): Trained ARIMA model.
    - test_data (array-like): Test data to evaluate the model on.
    
    Returns:
    - float: Root Mean Squared Error (RMSE) of the model on the test data.
    """
    # Make predictions on the test data
    predictions = model.forecast(steps=len(test_data))
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    
    return rmse
