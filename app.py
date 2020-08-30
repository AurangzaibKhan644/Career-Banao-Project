import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import Normalizer
import csv
from flask_sqlalchemy import SQLAlchemy 
from datetime import datetime


app = Flask(__name__)

# app.config['DATABASE_URL'] = 'postgres://hhxuorrwhgeiza:a7f9ade28be467c0735ac4f2071ee9b25444d319a59f424a3bcf6af69fe9229f@ec2-107-20-15-85.compute-1.amazonaws.com:5432/d6nueogb5lfg4h'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://hhxuorrwhgeiza:a7f9ade28be467c0735ac4f2071ee9b25444d319a59f424a3bcf6af69fe9229f@ec2-107-20-15-85.compute-1.amazonaws.com:5432/d6nueogb5lfg4h'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)


class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key=True)
    local_dt = db.Column(db.DateTime)
    
    # chotiya
    sch_percentage = db.Column(db.String(100))
    clg_percentage = db.Column(db.String(100))
    studying_hours = db.Column(db.String(100))
    extracurricular_activities = db.Column(db.String(100))
    competition = db.Column(db.String(100))
    scholarship = db.Column(db.String(100))
    communication_skills = db.Column(db.String(100))
    public_speaking_skills = db.Column(db.String(100))
    working_long_hours = db.Column(db.String(100))
    self_learning_capability = db.Column(db.String(100))
    extra_courses = db.Column(db.String(100))
    olympiad = db.Column(db.String(100))
    reading_writing_skills = db.Column(db.String(100))
    job_or_higher_studies = db.Column(db.String(100))
    sports = db.Column(db.String(100))
    technical_or_managerial = db.Column(db.String(100))
    hard_or_smart_worker = db.Column(db.String(100))
    teams = db.Column(db.String(100))
    introvert = db.Column(db.String(100))
    sch_major = db.Column(db.String(100))
    sch_fav_subject = db.Column(db.String(100))
    clg_major = db.Column(db.String(100))
    clg_fav_subject = db.Column(db.String(100))
    skills = db.Column(db.String(100))
    
    
    def __init__(self, customer, local_dt):
        self.customer = customer 
        self.local_dt = local_dt
       


# Loading ML Models
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

# Loading transformed state of encoders
working_hours_trans = pickle.load(open('working_hours.pkl', 'rb'))
int_sub_trans = pickle.load(open('int_sub.pkl', 'rb'))
workshop_trans = pickle.load(open('workshop.pkl', 'rb'))
alt_workshop_trans = pickle.load(open('alt_workshop.pkl', 'rb'))
cert_trans = pickle.load(open('cert.pkl', 'rb'))
alt_cert_trans = pickle.load(open('alt_cert.pkl', 'rb'))
uni_trans = pickle.load(open('uni.pkl', 'rb'))
alt_uni_trans = pickle.load(open('alt_uni.pkl', 'rb'))
job_role_trans = pickle.load(open('job_role.pkl', 'rb'))

# Loading Transformers 
comm_skills_trans = pickle.load(open('comm_skills.pkl', 'rb'))
speaking_skills_trans = pickle.load(open('speaking_skills.pkl', 'rb'))
self_learning_trans = pickle.load(open('self_learning_skills.pkl', 'rb'))
read_wri_skills_trans = pickle.load(open('read_wri_skills.pkl', 'rb'))
fav_sub_sch_trans = pickle.load(open('fav_sub_sch.pkl', 'rb'))
fav_sub_clg_trans = pickle.load(open('fav_sub_clg.pkl', 'rb'))
skills_trans = pickle.load(open('skills.pkl', 'rb'))



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
    working_hours_encoded = working_hours_trans.transform(working_hours)
    int_features.append(working_hours_encoded[0])
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
    
#     int_features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    int_features = request.get_json(force=True)
    
    # for server dataset
    sch_percentage_db = int_features[0]
    clg_percentage_db = int_features[1]
    studying_hours_db = int_features[2]
    extracurricular_activities_db = int_features[3]
    competition_db = int_features[4]
    scholarship_db = int_features[5]
    communication_skills_db = int_features[6]
    public_speaking_skills_db = int_features[7]
    working_long_hours_db = int_features[8]
    self_learning_capability_db = int_features[9]
    extra_courses_db = int_features[10]
    olympiad_db = int_features[11]
    reading_writing_skills_db = int_features[12]
    job_or_higher_studies_db = int_features[13]
    sports_db = int_features[14]
    technical_or_managerial_db = int_features[15]
    hard_or_smart_worker_db = int_features[16]
    teams_db = int_features[17]
    introvert_db = int_features[18]
    sch_major_db = int_features[19]
    sch_fav_subject_db = int_features[20]
    clg_major_db = int_features[21]
    clg_fav_subject_db = int_features[22]
    skills_db = int_features[23]
    
    
    
    data = [int_features[0], int_features[1], int_features[2]]
    normalized_data = Normalizer().fit_transform([data])
    
    int_features[0] = normalized_data[0][0]
    int_features[1] = normalized_data[0][1]
    int_features[2] = normalized_data[0][2]
    
    temp = comm_skills_trans.transform([int_features[6]])
    int_features[6] = temp[0]
    temp = speaking_skills_trans.transform([int_features[7]])
    int_features[7] = temp[0]
    temp = self_learning_trans.transform([int_features[9]])
    int_features[9] = temp[0]
    temp = read_wri_skills_trans.transform([int_features[12]])
    int_features[12] = temp[0]
    temp = fav_sub_sch_trans.transform([int_features[20]])
    int_features[20] = temp[0]
    temp = fav_sub_clg_trans.transform([int_features[22]])
    int_features[22] = temp[0]
    temp = skills_trans.transform([int_features[23]])
    int_features[23] = temp[0]
    
    final_features = [np.array(int_features)]
    working_hour = working_hours_model.predict(final_features)
    working_hours = int(working_hour[0])
    working_hours_encoded = working_hours_trans.transform(working_hour)
    int_features.append(working_hours_encoded[0])
    final_features = [np.array(int_features)]
    interested_subject = int_sub_model.predict(final_features)
    int_sub_encoded = int_sub_trans.transform(interested_subject)
    int_features.append(int_sub_encoded[0])
    final_features = [np.array(int_features)]
    workshop = workshops_model.predict(final_features)
    workshop_encoded = workshop_trans.transform(workshop)
    int_features.append(workshop_encoded[0])
    final_features = [np.array(int_features)]
    alt_workshop = alt_workshops_model.predict(final_features)
    alt_workshop_encoded = alt_workshop_trans.transform(alt_workshop)
    int_features.append(alt_workshop_encoded[0])
    final_features = [np.array(int_features)]
    certification = certification_model.predict(final_features)
    cert_encoded = cert_trans.transform(certification)
    int_features.append(cert_encoded[0])
    final_features = [np.array(int_features)]
    alt_certification = alt_certification_model.predict(final_features)
    alt_cert_encoded = alt_cert_trans.transform(alt_certification)
    int_features.append(alt_cert_encoded[0])
    final_features = [np.array(int_features)]
    university = uni_model.predict(final_features)
    uni_encoded = uni_trans.transform(university)
    int_features.append(uni_encoded[0])
    final_features = [np.array(int_features)]
    alt_university = alt_uni_model.predict(final_features)
    alt_uni_encoded = alt_uni_trans.transform(alt_university)
    int_features.append(alt_uni_encoded[0])
    final_features = [np.array(int_features)]
    career = career_model.predict(final_features)
    job_role_encoded = job_role_trans.transform(career)
    int_features.append(job_role_encoded[0])
    final_features = [np.array(int_features)]
    course = course_model.predict(final_features)
    
    # for server dataset
    working_hours_db = working_hours 
    interested_subject_db = interested_subject[0]
    workshop_db = workshop[0]
    alt_workshop_db = alt_workshop[0]
    certification_db = certification[0]
    alt_certification_db = alt_certification[0]
    university_db = university[0]
    alt_university_db = alt_university[0]
    career_db = career[0]
    course_db = course[0]

    customer = 'abcde'
    local_dt = datetime.now()
    data = Feedback(customer, local_dt)
    
    db.session.add(data)
    db.session.commit()
     
    
    return jsonify(working_hours=working_hours, interested_subject=interested_subject[0], workshop=workshop[0], alt_workshop=alt_workshop[0], certification=certification[0], alt_certification=alt_certification[0], university=university[0], alt_university=alt_university[0], career=career[0], course=course[0])


@app.route('/predict_api_temp',methods=['GET', 'POST'])
def predict_api_temp():
    '''
    For direct API calls trought request
    '''   
    data = request.get_json(force=True)
    working_hours = data[6]
    interested_subject = data[7]
    workshop = data[9]
    alt_workshop = data[12]
    certification = data[20]
    alt_certification = data[22]
    university = data[23]
    alt_university = data[6]
    career = data[7]
    course = data[9]
    
    return jsonify(working_hours=working_hours, interested_subject=interested_subject, workshop=workshop, alt_workshop=alt_workshop, certification=certification, alt_certification=alt_certification, university=university, alt_university=alt_university, career=career, course=course)


if __name__ == "__main__":
    app.run(debug=True)
