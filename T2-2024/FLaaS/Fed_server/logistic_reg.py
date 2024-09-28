import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from joblib import dump,load
import requests

class FederatedLogisticRegression:
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds
        self.model = None
        self.preprocessor = None
        
    def preprocess_data(self, X, y=None, fit=False):
        # Convert date to numerical format
        # print("Date in this >>>>>",X['date'])
        # X['date'] = pd.to_datetime(X['date'], format="%d-%m-%Y").map(pd.Timestamp.timestamp)
        # print("Date after transform in this >>>>>",X['date'])
         # Check if 'date' is already a Unix timestamp
        if X['date'].dtype == 'int64' or (X['date'].dtype == 'object' and X['date'].str.isnumeric().all()):
            # If it's already a timestamp, just ensure it's an int64
            X['date'] = X['date'].astype('int64')
        else:
            # If it's not a timestamp, assume it's in DD-MM-YYYY format and convert
            X['date'] = pd.to_datetime(X['date'], format='%d-%m-%Y')
            X['date'] = X['date'].astype('int64') // 10**9  # Convert to Unix timestamp


        if self.preprocessor is None or fit:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['date' ,'credit_amt', 'debit_amt', 'balance']),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), ['transaction_type'])
                ])
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = self.preprocessor.transform(X)
        
        if y is not None:
            return X_processed, y
        return X_processed
    
    def create_model(self):
        return LogisticRegression(warm_start=True)
    
    def train_local(self, X, y):
        if self.model is None:
            self.model = self.create_model()
        
        X_processed, y = self.preprocess_data(X, y, fit=True)
        self.model.fit(X_processed, y)
        
    def get_model_params(self):
        return self.model.coef_, self.model.intercept_
    
    def set_model_params(self, weights, intercept):
        self.model.coef_ = weights
        self.model.intercept_ = intercept
    
    def federated_learning(self, datasets):
        # Initialize the model with the first dataset
        X_init, y_init = datasets[0]
        self.train_local(X_init, y_init)
        
        for round in range(self.num_rounds):
            print(f"Round {round + 1}/{self.num_rounds}")
            
            global_weights = np.zeros_like(self.model.coef_)
            global_intercept = np.zeros_like(self.model.intercept_)
            
            for i, (X, y) in enumerate(datasets):
                print(f"Training on dataset {i + 1}/{len(datasets)}")
                self.train_local(X, y)
                weights, intercept = self.get_model_params()
                global_weights += weights
                global_intercept += intercept
            
            global_weights /= len(datasets)
            global_intercept /= len(datasets)
            
            self.set_model_params(global_weights, global_intercept)
    
    def predict(self, X):
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
    
    def save_model(self, filename):
        dump({'model': self.model, 'preprocessor': self.preprocessor}, filename)

    def get_model_weights(self):
        """
        Extract the weights and intercept of the model.
        
        :return: Dictionary containing weights and intercept
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return {
            'weights': self.model.coef_[0].tolist(),  # Convert numpy array to list
            'intercept': self.model.intercept_[0]
        }
    

def send_weights_to_api(weights, api_url):
    """
    Send the model weights to a specified API.
    
    :param weights: Dictionary containing weights and intercept
    :param api_url: String, URL of the API to send the weights to
    :return: API response
    """
    try:
        response = requests.post(api_url, json=weights)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending weights to API: {e}")
        return None

def federated_learning_service(data, target_column="is_split", num_rounds=5):
    """
    :param data: List of pandas DataFrames containing the datasets
    :param target_column: String, name of the target column
    :param num_rounds: Integer, number of federated learning rounds
    :return: Tuple containing the trained model and test accuracy
    """
    api_url = "http://127.0.0.1:5005/receive_weights"

    # Prepare datasets
    datasets = [(df.drop(columns=[target_column]), df[target_column]) for df in data]
    
    # Initialize and train the federated model
    fed_model = FederatedLogisticRegression(num_rounds=num_rounds)
    fed_model.federated_learning(datasets)
    
    # Save the trained model
    # model_filename = 'federated_logistic_model.joblib'
    # fed_model.save_model(model_filename)
    # print(f"Model saved as '{model_filename}'")
    
    # Evaluate the model on the last dataset (assuming it's the test set)
    X_test, y_test = datasets[-1]
    predictions = fed_model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy:.2f}")

    # extracting model weights
    model_weights = fed_model.get_model_weights()

    print("The model weights are>>>> \n",model_weights)
    print(">>>>>>>>>>>>>>>>>>>>>")
    # Send weights to API if URL is provided
    api_response = None
    if api_url:
        api_response = send_weights_to_api(model_weights, api_url)
        if api_response:
            print("Weights successfully sent to API")
        else:
            print("Failed to send weights to API")
    
    return fed_model, accuracy
    # return api_response


def prediction_service(prediction_data):
    """
    
    
    :param prediction_data: pandas DataFrame containing the data for prediction
    :return: numpy array of predictions
    """
    # Load the saved model
    model_filename = './federated_logistic_model.joblib'
    try:
        loaded_model = load(model_filename)
        fed_model = FederatedLogisticRegression()
        fed_model.model = loaded_model['model']
        fed_model.preprocessor = loaded_model['preprocessor']
    except FileNotFoundError:
        raise Exception(f"Model file '{model_filename}' not found. Please train the model first.")

    # Make predictions
    predictions = fed_model.predict(prediction_data)
    
    return predictions
