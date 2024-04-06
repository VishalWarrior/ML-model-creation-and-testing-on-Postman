from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

# Load the pickled model
# model = pickle.load(open("model.pkl", "rb"))
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    print (data)
    
    # Convert the dict data to a numpy array
    predict_data = scaler.transform(np.array([data['Sepal_Length'], data['Sepal_Width'], data['Petal_Length'], data['Petal_Width']]).reshape(1, -1))
    
    
    # Make prediction
    prediction = model.predict(predict_data)
    
    # Return the prediction as JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
