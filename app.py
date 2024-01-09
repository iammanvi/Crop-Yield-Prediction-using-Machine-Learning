from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

#loding modles
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Year = request.form['Year'].strip()
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year'].strip()
        pesticides_tonnes = request.form['pesticides_tonnes'].strip()
        avg_temp = request.form['avg_temp'].strip()
        Area = request.form['Area'].strip()
        Item = request.form['Item'].strip()

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])

        transformed_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html',predicted_value=predicted_value)



# python main
if __name__=='__main__':
    app.run(debug=True)

