from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def train_and_evaluate_model(data, model_type='additive'):
    """
    Train and evaluate a Decomposition model on the time series data.
    
    Parameters:
    - data (DataFrame): Pandas DataFrame containing the time series data.
    - model_type (str): Type of decomposition model: 'additive' or 'multiplicative'.
    
    Returns:
    - tuple: Tuple containing the trend, seasonal, and residual components.
    """
    try:
        # Perform decomposition
        decomposition = seasonal_decompose(data['value'], model=model_type)
        
        # Visualize decomposition
        visualize_decomposition(decomposition)
        
        return decomposition.trend, decomposition.seasonal, decomposition.resid
    except Exception as e:
        print("An error occurred during model training and evaluation:", e)
        return None, None, None

def visualize_decomposition(decomposition):
    """
    Visualize the decomposition components.
    
    Parameters:
    - decomposition: Result of seasonal decomposition.
    """
    try:
        plt.figure(figsize=(10, 8))
        
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('Original Time Series')
        
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal')
        
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('Residual')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("An error occurred during visualization:", e)
