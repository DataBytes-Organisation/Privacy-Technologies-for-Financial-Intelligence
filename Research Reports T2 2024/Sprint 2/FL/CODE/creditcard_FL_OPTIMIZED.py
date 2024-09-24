from typing import Dict, List, Tuple
import tensorflow as tf
import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import logging
from flwr.common import Context
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import flwr

VERBOSE = 0

from sklearn.preprocessing import LabelEncoder

# Preprocessing function specific to creditcard.csv
def preprocess_creditcard_dataset():
    dataset_df = pd.read_csv("creditcard.csv")  # Adjust the file path if necessary

    # Use SMOTE to balance the dataset (if necessary)
    def balance_classes_with_smote(data_df):
        features = data_df.drop(columns=["isFraud"])  # Assuming 'isFraud' is the column to predict
        labels = data_df["isFraud"]

        # Apply SMOTE to oversample the minority class
        smote = SMOTE(random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        return pd.DataFrame(features_resampled), pd.Series(labels_resampled)

    # Drop unnecessary or high cardinality columns
    selected_columns = ['amount', 'type', 'isFraud']  # Add or remove columns as necessary
    dataset_df_red = dataset_df[selected_columns]

    # Apply label encoding for high cardinality columns (if needed)
    label_encoder = LabelEncoder()
    if 'type' in dataset_df_red.columns:
        dataset_df_red['type'] = label_encoder.fit_transform(dataset_df_red['type'])  # Label encode 'type' column

    # Apply SMOTE to handle class imbalance
    features_balanced, labels_balanced = balance_classes_with_smote(dataset_df_red)

    # Scale the features
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features_balanced), columns=features_balanced.columns)

    # Combine features and labels back
    dataset_df_scaled = pd.concat([features_scaled, labels_balanced], axis=1)

    return dataset_df_scaled

# Preprocess the creditcard.csv dataset
dataset_df_scaled = preprocess_creditcard_dataset()

# Train-test split
train_df, test_df = train_test_split(dataset_df_scaled, test_size=0.1, random_state=42)

# Convert to numpy arrays
train_features = train_df.drop(columns=["isFraud"]).values
train_labels = train_df["isFraud"].values
test_features = test_df.drop(columns=["isFraud"]).values
test_labels = test_df["isFraud"].values

# Combine features and labels for the training dataset
train_data = np.concatenate((train_features, train_labels.reshape(-1, 1)), axis=1)

# Combine features and labels for the test dataset
test_data = np.concatenate((test_features, test_labels.reshape(-1, 1)), axis=1)

# Ensure that data is in float32 format for TensorFlow compatibility
train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)

# Create Partitions for federated learning
num_partitions = 10
partitions = []
partition_size = len(train_data) // num_partitions

for i in range(num_partitions):
    start_idx = i * partition_size
    end_idx = (i + 1) * partition_size
    partition_data = train_data[start_idx:end_idx]
    partitions.append(partition_data)

NUM_CLIENTS = num_partitions

# Number of samples in train and test data
num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]
print(f"Number of samples in train data: {num_train_samples}")
print(f"Number of samples in test data: {num_test_samples}")

# Define the model
def get_model():
    """Construct a simple binary classification model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(train_data.shape[1] - 1,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# FlowerClient class
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset) -> None:
        self.model = get_model()
        self.trainset = trainset
        self.valset = valset

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        train_features = self.trainset[:, :-1]
        train_labels = self.trainset[:, -1]
        self.model.fit(train_features, train_labels, epochs=1, verbose=VERBOSE)
        return self.model.get_weights(), len(train_features), {}

def evaluate(self, parameters, config):
    self.model.set_weights(parameters)
    val_features = self.valset[:, :-1]  # Extract features
    val_labels = self.valset[:, -1]  # Extract labels

    # Get predictions (in binary format for classification)
    predictions = self.model.predict(val_features)
    predicted_labels = (predictions > 0.5).astype(int)

    # Calculate accuracy
    loss, acc = self.model.evaluate(val_features, val_labels, verbose=0)

    # Calculate precision, recall, and F1-score
    precision = precision_score(val_labels, predicted_labels)
    recall = recall_score(val_labels, predicted_labels)
    f1 = f1_score(val_labels, predicted_labels)

    # Print a classification report for detailed metrics
    print(classification_report(val_labels, predicted_labels))

    # Return loss and metrics
    return loss, len(val_features), {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metrics (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples)
    }

# Function to create clients
def get_client_fn(partitions: List[np.ndarray], testset: np.ndarray):
    def client_fn(cid: str) -> fl.client.Client:
        partition = partitions[int(cid)]
        trainset, valset = partition, testset
        return FlowerClient(trainset, valset)
    return client_fn

def get_evaluate_fn(testset: np.ndarray):
    def evaluate(server_round: int, parameters: fl.common.NDArray, config: Dict[str, fl.common.Scalar]):
        model = get_model()
        model.set_weights(parameters)
        val_features = testset[:, :-1]
        val_labels = testset[:, -1]
        loss, accuracy = model.evaluate(val_features, val_labels, verbose=VERBOSE)

        # Add additional metrics calculations here
        predictions = (model.predict(val_features) > 0.5).astype(int)
        precision = precision_score(val_labels, predictions)
        recall = recall_score(val_labels, predictions)
        f1 = f1_score(val_labels, predictions)

        # Return aggregated metrics
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    return evaluate

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,
    fraction_evaluate=0.05,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=int(NUM_CLIENTS * 0.75),
    evaluate_fn=get_evaluate_fn(test_data),  # This should include precision, recall, and F1-score
    # Define custom evaluation metric aggregation
    evaluate_metrics_aggregation_fn=weighted_average
)

# Define the resources each client should use
client_resources = {
    "num_cpus": 1,  # Allocate 1 CPU per client (adjust based on your system)
    "num_gpus": 0.1  # Allocate 10% of a GPU per client (if you're using GPUs)
}


history = fl.simulation.start_simulation(
    client_fn=get_client_fn(partitions, test_data),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources=client_resources
)

# After training, you'll have the aggregated precision, recall, and F1-score
print("Final metrics:", history)
