import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

working_hours_model = pickle.load(open('working_hours_model.pkl', 'rb'))
int_sub_model = pickle.load(open('interested_subject_model.pkl', 'rb'))
workshops_model = pickle.load(open('workshops_model.pkl', 'rb'))
alt_workshops_model = pickle.load(open('alternate_workshops_model.pkl', 'rb'))
certification_model = pickle.load(open('certifications_model.pkl', 'rb'))
alt_certification_model = pickle.load(open('alternate_certifications_model.pkl', 'rb'))
uni_model = pickle.load(open('university_model.pkl', 'rb'))
alt_uni_model = pickle.load(open('alternate_university_model.pkl', 'rb'))
career_model = pickle.load(open('career_model.pkl', 'rb'))
course_model = pickle.load(open('course_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
#   int_features = [int(x) for x in request.form.values()] 
    int_features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#   int_features = [1, 1, 1, 0, 0, 1, 2, 2, 0, 3, 0, 0, 1, 5, 0, 9, 1, 7, 0, 0, 1, 1, 0, 11, 5, 7, 42, 21, 4, 0, 0, 1, 1]
    final_features = [np.array(int_features)]
    working_hours = working_hours_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    int_subject = int_sub_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    workshop = workshops_model.predict(final_features)
    int_features.append(1)
    int_features.append(1)
    final_features = [np.array(int_features)]
    certification = certification_model.predict(final_features)
    int_features.append(1)
    int_features.append(1)
    final_features = [np.array(int_features)]
    university = uni_model.predict(final_features)
    int_features.append(1)
    int_features.append(1)
    final_features = [np.array(int_features)]
    career = career_model.predict(final_features)

#return render_template('index.html', working_hours='Predicted Working Hours : {}'.format(working_hours[0]), int_subject='Predicted Interested Subject : {}'.format(int_subject[0]), workshop='Predicted University : {}'.format(workshop[0]), certification='Predicted University : {}'.format(certification[0]), university='Predicted University : {}'.format(university[0]), career='Predicted University : {}'.format(career[0])
    return render_template('index.html', working_hours='Predicted Working Hours : {}'.format(working_hours[0]), int_subject='Predicted Interested Subjects : {}'.format(int_subject[0]), workshop='Predicted Workshop : {}'.format(workshop[0]), certification='Predicted Certification : {}'.format(certification[0]), university='Predicted University : {}'.format(university[0]), career='Predicted Career : {}'.format(career[0]))
    
    

@app.route('/predict_api',methods=['GET', 'POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''   
#     data = request.get_json(force=True)
#     working_hours = data['working_hours']
#     interested_subject = data['interested_subject']
#     workshop = data['workshop']
#     certification = data['certification']
#     university = data['university']
#     career = data['career']
    
    int_features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    final_features = [np.array(int_features)]
    working_hour = working_hours_model.predict(final_features)
    working_hours = int(working_hour[0])
    int_features.append(1)
    final_features = [np.array(int_features)]
    interested_subject = int_sub_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    workshop = workshops_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    alt_workshop = alt_workshops_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    certification = certification_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    alt_certification = alt_certification_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    university = uni_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    alt_university = alt_uni_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    career = career_model.predict(final_features)
    int_features.append(1)
    final_features = [np.array(int_features)]
    course = course_model.predict(final_features)
    
    return jsonify(working_hours=working_hours, interested_subject=interested_subject[0], workshop=workshop[0], alt_workshop=alt_workshop[0], certification=certification[0], alt_certification=alt_certification[0], university=university[0], alt_university=alt_university[0], career=career[0], course=course[0])


if __name__ == "__main__":
    app.run(debug=True)
