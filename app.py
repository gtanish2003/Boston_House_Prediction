from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

application=Flask(__name__)
app=application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            crim=request.form.get('CRIM'),
            zn=request.form.get('ZN'),
            indus=request.form.get('INDUS'),
            chas=float(request.form.get('CHAS')),
            nox=request.form.get('NOX'),
            rm=request.form.get('RM'),
            age=request.form.get('AGE'),
            dis=request.form.get('DIS'),
            rad=request.form.get('RAD'),
            tax=request.form.get('TAX'),
            ptratio=request.form.get('PTRATIO'),
            b=request.form.get('B'),
            lstat=request.form.get('LSTAT')

         
        )

    pred_df=data.get_data_as_data_frame()
    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)

    return render_template('home.html',results=results[0])
    


if __name__=='__main__':
    app.run(debug=True)
