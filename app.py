import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

 #working_hours_model = pickle.load(open('working_hours_model.pkl', 'rb'))
 #int_sub_model = pickle.load(open('int_sub_model.pkl', 'rb'))
# workshops_model = pickle.load(open('workshops_model.pkl', 'rb'))
cert_model = pickle.load(open('cert_model.pkl', 'rb'))
 #uni_model = pickle.load(open('uni_model.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
#    int_features = [int(x) for x in request.form.values()] 
    #int_features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    int_features = [1, 1, 1, 0, 0, 1, 2, 2, 0, 3, 0, 0, 1, 5, 0, 9, 1, 7, 0, 0, 1, 1, 0, 11, 5, 7, 42, 21, 4, 0, 0]
    final_features = [np.array(int_features)]
    prediction = cert_model.predict(final_features)
    #final_features = [np.array(int_features)]
    #prediction = working_hours_model.predict(final_features)
    
    #int_features.append(1)
    #final_features = [np.array(int_features)]
    #prediction2 = int_sub_model.predict(final_features)
    #int_features.append(1)
#     final_features = [np.array(int_features)]
#     prediction2 = workshops_model.predict(final_features)
    #int_features.append(1)
#     final_features = [np.array(int_features)]
#     prediction3 = cert_model.predict(final_features)
    #int_features.append(1)
    #final_features = [np.array(int_features)]
    #prediction4 = uni_model.predict(final_features)
#     int_features.append(1)
#     final_features = [np.array(int_features)]
#     prediction5 = model.predict(final_features)

    #return render_template('index.html', prediction_text='Predicted Working Hours : {}'.format(prediction[0]), prediction_text2='Predicted Interested Subject : {}'.format(prediction2[0]), prediction_text4='Predicted University : {}'.format(prediction4[0]))
    return render_template('index.html', prediction_text='Certifications : {}'.format(prediction[0]))
    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''   
    #data = request.get_json(force=True)
    int_features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
#    output = prediction[0]
    return jsonify(subject=prediction[0], certification=prediction3[0], university=prediction4[0])


if __name__ == "__main__":
    app.run(debug=True)
