from flask import Flask, request, jsonify, send_file
from logistic_reg import FederatedLogisticRegression, federated_learning_service, prediction_service
import pandas as pd
import numpy as np
from joblib import load
import io
import csv

app = Flask(__name__)

# Initialize the model
fed_model = FederatedLogisticRegression(num_rounds=5)

@app.route('/train', methods=['POST'])
def train_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Read the CSV file
        # content = file.read()
        # data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        data = pd.read_csv(file)
        # print(data.head())
        fed_model, accuracy, api_response = federated_learning_service([data])
        # resp = federated_learning_service([data])
        
        return jsonify({'message': 'Model trained successfully'}), 200
        # return api_response, 200


@app.route('/predict', methods=['POST'])
def predict():
    # if not fed_model.model:
    #     return jsonify({'error': 'Model not trained yet'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Read the CSV file
        # content = file.read()
        data = pd.read_csv(file)
        
        # Make predictions
        predictions = prediction_service(data)
        
        # Prepare the results
        result = pd.DataFrame({'prediction': predictions})
        result_csv = result.to_csv(index=False)
        
        # Create an in-memory file-like object
        mem = io.StringIO()
        mem.write(result_csv)
        mem.seek(0)
        
        # Return the CSV file
        return send_file(
            io.BytesIO(mem.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='predictions.csv'
        )

@app.route('/save_model', methods=['POST'])
def save_model():
    if not fed_model.model:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    filename = request.json.get('filename', 'federated_logistic_model.joblib')
    fed_model.save_model(filename)
    
    return jsonify({'message': f'Model saved as {filename}'}), 200

@app.route('/load_model', methods=['POST'])
def load_model():
    filename = request.json.get('filename', 'federated_logistic_model.joblib')
    try:
        loaded = load(filename)
        fed_model.model = loaded['model']
        fed_model.preprocessor = loaded['preprocessor']
        return jsonify({'message': f'Model loaded from {filename}'}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Model file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True,port=5000)