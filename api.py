from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = load('knn.joblib')
scaler = load('scaler.joblib')
one_hot_columns = [
    "Health & Fitness",
    "Personal Development",
    "Productivity & Time Management",
    "Social & Relationships",
    "Financial Habits",
    "Environmental Impact",
    "Emotional Well-being",
    "Work & Career",
    "Creative & Hobbies",
    "Spirituality & Reflection"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
            return jsonify({'error': 'No JSON data received'}), 400
    
    
    start_date = data.get('StartDate')
    type_ = data.get('Type')
    curr_streak = data.get('CurrStreak')

    if not start_date or not type_ or curr_streak is None:
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        start_date = pd.to_datetime(start_date)
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    start_date_timestamp = (start_date - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

    
    input_data = pd.DataFrame({
        'StartDate': [start_date_timestamp],
        'Type': [type_],
        'CurrStreak': [curr_streak]
    })

    input_data = scaler.fit_transform(input_data)
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        return False
    else:
        return True

if __name__ == '__main__':
    app.run(debug=True)