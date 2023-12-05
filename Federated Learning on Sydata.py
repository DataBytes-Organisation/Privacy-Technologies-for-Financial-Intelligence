# Code for Federated Learning (FL) using TensorFlow
import tensorflow as tf
import numpy as np

# Define the client model
def client_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Generate synthetic training data
num_clients = 10  # Number of simulated clients
num_samples_per_client = 100  # Number of training samples per client
input_shape = (784,)  # Shape of the input data (e.g., for MNIST images)

# Initialize empty lists for training data and labels
train_data = []
train_labels = []

for i in range(num_clients):
    # Generate random training data for each client
    X = np.random.normal(loc=0, scale=1, size=(num_samples_per_client, input_shape[0]))
    y = np.random.randint(0, 10, size=num_samples_per_client)

    # Append the generated data and labels to the respective lists
    train_data.append(X)
    train_labels.append(y)

# Initialize the global model
global_model = client_model(input_shape)

# Perform federated learning rounds
num_rounds = 10

for round in range(num_rounds):
    # Update local models on each client
    local_updates = []

    for i in range(num_clients):
        # Extract local training data and labels
        X = train_data[i]
        y = train_labels[i]

        # Train the local model on client data
        local_model = client_model(input_shape)
        local_model.fit(X, y, epochs=1, batch_size=32)

        # Compute local model updates
        local_updates.append(local_model.get_weights())

    # Aggregate local model updates into the global model
    global_model.set_weights(tf.math.reduce_mean(local_updates, axis=0))

# Evaluate the global model on the entire dataset
X_eval = np.concatenate(train_data)
y_eval = np.concatenate(train_labels)

loss, accuracy = global_model.evaluate(X_eval, y_eval)

print("Global model evaluation loss:", loss)
print("Global model evaluation accuracy:", accuracy)

