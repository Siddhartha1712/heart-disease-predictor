from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

heart_data = pd.read_csv('heart.csv')
X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def get_risk_result(age, chol, trestbps, target):
    if target == 1:
        if age > 50 and chol > 240:
            return "The person has a High risk heart disease."
        elif trestbps > 140:
            return "The person has a Moderate risk heart disease."
        else:
            return "The person has a Mild risk heart disease."
    else:
        
        return "The person has No heart disease detected."
    
def get_treatment_plan(age, chol, trestbps, target):
    if target == 1:
        if age > 50 and chol > 240:
            return "Consult for Heart treatment is recommended."
        elif trestbps > 140:
            return "Consult for blood pressure management is recommended."
        else:
            return "consult for heart health."
    else:
        return "Maintain the diet and health."
    

def get_precaution_plan(age, chol, trestbps, target):
    if target == 1:
        if age > 50 and chol > 240:
            return "Take prescribed medicines, eat a heart-healthy diet, exercise regularly, \nand have regular check-ups.\nWhat to Avoid: Stay away from unhealthy foods, smoking, and excessive alcohol.\nAdditional Tips: Manage stress and get 7-9 hours of sleep per night.\n\n"

        elif trestbps > 140:
            return "\nManage blood pressure, eat a balanced diet, exercise regularly,\nand maintain a healthy weight.\nWhat to Avoid: Avoid too much salt, long periods of sitting, and unhealthy snacks.\nAdditional Tips: Stay hydrated and manage stress effectively.\n"
        else:
            return "Keep up with regular check-ups, maintain a healthy diet and exercise routine,\nand manage stress.\nWhat to Avoid: Avoid overeating, inactivity, and ignoring new symptoms.\nAdditional Tips: Stay informed about heart health and consider joining support groups.\n\n"

    else:
        return  "Maintain a balanced diet, regular exercise, and stress management.\nWhat to Avoid: Don’t start smoking, excessive drinking, or neglect health check-ups.\nAdditional Tips: Continue following preventive measures and maintain a positive outlook.\n\n"



@app.route('/', methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            chol = int(request.form['cholesterol'])
            trestbps = int(request.form['restBps'])
            thalach = int(request.form['maxhr'])
            fbs = int(request.form['fastBps'])
            cp = int(request.form['cp'])
            oldpeak = float(request.form['std'])
            slope = int(request.form['slope'])
            ca = int(request.form['no_v'])
            thal = int(request.form['thal'])
            excang = int(request.form['excang'])
            restecg = int(request.form['ecg'])

            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, excang, oldpeak, slope, ca, thal]])

            if input_data.shape[1] != len(X.columns):
                return "Error: Input data does not match model's expected feature size."

            input_df = pd.DataFrame(input_data, columns=X.columns)

            prediction = model.predict(input_df)[0]

            risk = get_risk_result(age, chol, trestbps, prediction)

            treatment_plan = get_treatment_plan(age,chol,trestbps,prediction)

            precaution = get_precaution_plan(age,chol,trestbps,prediction)

            return render_template("output.html", age=age, sex=sex, cholesterol=chol, restBps=trestbps, max_hr=thalach,
                                   fastBps=fbs, cp=cp, std=oldpeak, slope=slope, no_vessel=ca, thal=thal,
                                   excang=excang, ecg=restecg, prediction=prediction, risk=risk,treatment_plan=treatment_plan,precaution=precaution)
        
        except Exception as e:
            return str(e) 

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
