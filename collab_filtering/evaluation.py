import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_df):
    """
    Evaluate the model on test data
    
    test_df: DataFrame with columns ['Movie-id', 'User-id', 'rating']
    """
    print("\n=== MODEL EVALUATION ===")

    predictions = model.predict_batch(test_df)
    actual = test_df['rating'].values

    # Metrics
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Error analysis
    errors = predictions - actual
    print(f"\nMean Error: {errors.mean():.4f}")
    print(f"Error Std Dev: {errors.std():.4f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'predictions': predictions}