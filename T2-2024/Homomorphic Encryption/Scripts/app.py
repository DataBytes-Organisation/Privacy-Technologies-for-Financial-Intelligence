from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # To load the pre-trained model

app = Flask(__name__)

# Load the pre-trained model (ensure you have saved it after training)
model = joblib.load('model.pkl')

# Assuming you have a scaler that was used for feature scaling
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # This will render your HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the incoming data (sent from the frontend form)
    input_data = request.get_json()

    # Extract the features from the input data
    transaction_amount = float(input_data['Transaction_Amount_AUD'])
    credit_score = int(input_data['Credit_Score'])
    age = int(input_data['Age'])
    transaction_type = input_data['Transaction_Type']  # e.g., 'Online', 'In-Person'
    time_of_day = input_data['Time_of_Day']  # e.g., 'Morning', 'Evening'

    # Use the same data structure for prediction as the training data
    input_features = np.array([[transaction_amount, credit_score, age, transaction_type, time_of_day]])

    # Standardize the input features using the same scaler from training
    input_features_scaled = scaler.transform(input_features)

    # Make prediction using the pre-trained model
    prediction = model.predict(input_features_scaled)
    prediction_prob = model.predict_proba(input_features_scaled)[:, 1]  # Probability of fraud (class 1)

    # Prepare response data
    response_data = {
        "prediction": int(prediction[0]),
        "fraud_probability": float(prediction_prob[0])
        # Add more fields if necessary (like roc_auc, classification_report, etc.)
    }

    # Return prediction and other model metrics as JSON
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
