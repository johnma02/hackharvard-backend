from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flasgger import Swagger
from datetime import datetime

crime_model = None


app = Flask(__name__)
swagger = Swagger(app)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/predict_crime', methods=['POST'])
def predict_crime():
    """Endpoint returning a classification and regression value for crime prediction based on location and time data.
    ---
    parameters:
      - in: body
        name: input_data
        description: JSON input data for prediction.
        required: true
        schema:
          type: object
          properties:
            lat:
              type: number
              format: float
              description: Latitude.
            lng:
              type: number
              format: float
              description: Longitude.
            date:
              type: string
              format: date-time
              description: Date and time.

    responses:
      200:
        description: Classification and regression values for crime prediction based on location and time data.
        schema:
          type: object
          properties:
            prediction:
              type: number
              description: Classification value.
            regression:
              type: number
              format: float
              description: Regression value.
        examples:
          application/json:
            {
              "prediction": 3,
              "regression": 3.14159
            }
    """
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    date = datetime.fromisoformat(data.get('date'))
    with open('./models/crime_model_xgboost.pkl', 'rb') as model_file:
        crime_model = pickle.load(model_file)
    prediction = None
    if crime_model is not None:
        prediction_input = pd.DataFrame({
            'Lat': [lat],
            'Long': [lng],
            'hour_sin': [np.sin(2 * np.pi * date.hour / 24.0)],
            'hour_cos': [np.cos(2 * np.pi * date.hour / 24.0)],
            'month_sin': [np.sin(2 * np.pi * date.month / 12.0)],
            'month_cos': [np.cos(2 * np.pi * date.month / 12.0)],
        })
        prediction = crime_model.predict(prediction_input)
    threshholds = [0, 1, 2, 3, 4, 5]
    return jsonify({
        "prediction": np.digitize(prediction, threshholds).tolist()[0],
        "regression": prediction.tolist()[0]},
        200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
