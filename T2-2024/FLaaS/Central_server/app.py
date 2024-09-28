from flask import Flask, request, jsonify, send_file
from modelUpdateService import update_logistic_regression_weights
import pandas as pd
import numpy as np
from joblib import load
import io
import csv

app = Flask(__name__)
# Global variables to store weights and intercept
global_weights = None
global_intercept = None

@app.route('/receive_weights', methods=['POST'])
def receive_weights():
    global global_weights, global_intercept
    
    try:
        data = request.get_json()
        
        if 'weights' not in data or 'intercept' not in data:
            return jsonify({"error": "Missing 'weights' or 'intercept' in the payload"}), 400
        
        # Store weights and intercept in global variables
        global_weights = np.array(data['weights'])
        global_intercept = data['intercept']
        
        update_logistic_regression_weights(global_weights,global_intercept)

        return jsonify({
            "message": "Weights and intercept received successfully",
            "weights_shape": global_weights.shape,
            "intercept": global_intercept
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_weights', methods=['GET'])
def get_weights():
    if global_weights is None or global_intercept is None:
        return jsonify({"error": "Weights and intercept have not been set yet"}), 404
    
    return jsonify({
        "weights": global_weights.tolist(),
        "intercept": global_intercept
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port = 5005)