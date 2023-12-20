from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
transformer = pickle.load(open("models/transformer.pkl","rb"))
pipeline = pickle.load(open("models/pipe.pkl","rb"))



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods =["POST","GET"])
def predict():


        if request.method == "POST":
                education = request.form.get("Education")
                joining_year = int(request.form.get("Joining_Year"))
                city = request.form.get("City")
                payment_tier = int(request.form.get("Payment_Tier"))
                age = int(request.form.get("Age"))
                gender = request.form.get("Gender")
                benched = request.form.get("benched")
                experience = int(request.form.get("Experience"))

                input_data =[
                        education,
                        joining_year,
                        city,
                        payment_tier,
                        age,
                        gender,
                        benched,
                        experience
                        ]
                
        input_data =(np.array([input_data],dtype=object)).reshape((1,-1))
                
        columns = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender',
       'EverBenched', 'ExperienceInCurrentDomain']
        
        data = pd.DataFrame(input_data,columns=columns)

        transformed_data = transformer.transform(data)
        prediction = pipeline.predict(transformed_data)

        out = 'Employee will leave' if prediction == 1 else 'Employee will stay'
        print(prediction,out)                    
        return render_template("index.html",out=out)

if __name__=="__main__":
        app.run()
    