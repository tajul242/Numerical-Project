from flask import Flask, request, jsonify
from flask_cors import CORS
from regression import (
    parse_data, linear_regression, polynomial_regression,
    exponential_regression, logarithmic_regression
)
import json

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.get_json()
    method = data.get('method', 'linear')
    raw_data = data.get('data', '')
    
    points = parse_data(raw_data)
    
    if len(points) < 2:
        return jsonify({"error": "Need at least 2 data points"}), 400
    
    try:
        if method == 'linear':
            result = linear_regression(points)
        elif method == 'polynomial':
            degree = data.get('degree', 2)
            if degree >= len(points):
                return jsonify({"error": "Degree must be less than data points"}), 400
            result = polynomial_regression(points, degree)
        elif method == 'exponential':
            if any(p[1] <= 0 for p in points):
                return jsonify({"error": "All y values must be positive"}), 400
            result = exponential_regression(points)
        elif method == 'logarithmic':
            if any(p[0] <= 0 for p in points):
                return jsonify({"error": "All x values must be positive"}), 400
            result = logarithmic_regression(points)
        else:
            return jsonify({"error": "Unknown method"}), 400
        
        # Convert predict function to string for JSON
        response = {
            "method": result["method"],
            "equation": result["equation"],
            "coefficients": result["coefficients"],
            "steps": result["steps"],
            "graph": result["graph"]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    method = data.get('method')
    coeffs = data.get('coefficients')
    x_val = data.get('x')
    
    if method == 'linear':
        y = coeffs['m'] * x_val + coeffs['c']
    elif method == 'polynomial':
        y = sum(coeffs[i] * (x_val ** i) for i in range(len(coeffs)))
    elif method == 'exponential':
        y = coeffs['a'] * (2.718281828459045 ** (coeffs['b'] * x_val))
    elif method == 'logarithmic':
        import math
        y = coeffs['a'] * math.log(x_val) + coeffs['b']
    else:
        return jsonify({"error": "Unknown method"}), 400
    
    return jsonify({"x": x_val, "y": y})

@app.route('/')
def home():
    return "Curve Fitting Solver API - Running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)