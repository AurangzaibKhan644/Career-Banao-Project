import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

int_sub_model = pickle.load(open('int_sub_model.pkl', 'rb'))
# workshops_model = pickle.load(open('workshops_model.pkl', 'rb'))
cert_model = pickle.load(open('cert_model.pkl', 'rb'))
uni_model = pickle.load(open('uni_model.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] 
    final_features = [np.array(int_features)]
    prediction = int_sub_model.predict(final_features)
    int_features.append(1)
#     final_features = [np.array(int_features)]
#     prediction2 = workshops_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    prediction3 = cert_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    prediction4 = uni_model.predict(final_features)
#     int_features.append(1)
#     final_features = [np.array(int_features)]
#     prediction5 = model.predict(final_features)

    return render_template('index.html', prediction_text='Predicted Interested Subject : {}'.format(prediction[0]), prediction_text3='Predicted Certification : {}'.format(prediction3[0]), prediction_text4='Predicted University : {}'.format(prediction4[0]))



if __name__ == "__main__":
    app.run(debug=True)
