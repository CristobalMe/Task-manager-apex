from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

loaded_model = pickle.load(open('svm_model.sav', 'rb'))

loaded_pre_processing = pickle.load(open('pre_processing.sav', 'rb'))


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

    prediction = loaded_model.predict(input_data)
    
    if prediction == 1:
        return jsonify({'notification': False}), 200
    else:
        return jsonify({'notification': True}), 200

if __name__ == '__main__':
    app.run(debug=True)