import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

def update_logistic_regression_weights( new_weights, new_intercept):
    """
    Update the weights and intercept of a logistic regression model,
    handling potential version mismatches and dictionary format.
    
    Args:
    new_weights (array-like): New weights for the model. Should match the shape of the original weights.
    new_intercept (float or array-like): New intercept for the model.
    
    Returns:
    dict: Updated model data in dictionary format.
    """

    model_path = "./federated_logistic_model.joblib"
    try:
        # Attempt to load the model, suppressing warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model_data = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new model data structure with the provided weights and intercept.")
        model_data = {}

    # Convert new_weights and new_intercept to numpy arrays
    new_weights = np.array(new_weights)
    new_intercept = np.array(new_intercept)

    # Update weights
    if 'coef_' in model_data:
        if model_data['coef_'].shape != new_weights.shape:
            print(f"Warning: Shape mismatch. Reshaping new_weights from {new_weights.shape} to {model_data['coef_'].shape}")
            model_data['coef_'] = new_weights.reshape(model_data['coef_'].shape)
        else:
            model_data['coef_'] = new_weights
    else:
        model_data['coef_'] = new_weights

    # Update intercept
    if 'intercept_' in model_data:
        if model_data['intercept_'].shape != new_intercept.shape:
            print(f"Warning: Intercept shape mismatch. Reshaping new_intercept from {new_intercept.shape} to {model_data['intercept_'].shape}")
            model_data['intercept_'] = new_intercept.reshape(model_data['intercept_'].shape)
        else:
            model_data['intercept_'] = new_intercept
    else:
        model_data['intercept_'] = new_intercept

    # Save the updated model data
    joblib.dump(model_data, model_path)
    
    return model_data



def create_logistic_regression_from_dict(model_data):
    """
    Create a LogisticRegression object from a dictionary of model data.
    
    Args:
    model_data (dict): Dictionary containing 'coef_' and 'intercept_' keys.
    
    Returns:
    sklearn.linear_model.LogisticRegression: A LogisticRegression object with the specified coefficients and intercept.
    """
    model = LogisticRegression()
    model.coef_ = model_data['coef_']
    model.intercept_ = model_data['intercept_']
    return model

