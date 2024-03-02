import sys
import os

import numpy as np 
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:

            
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)

            return preds
        

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
            self,
            crim,
            zn,
            indus,
            chas,
            nox,
            rm,
            age,
            dis,
            rad,
            tax,
            ptratio,
            b,
            lstat
    ):
        self.crim=crim
        self.zn=zn
        self.indus=indus
        self.chas=chas
        self.nox=nox
        self.rm=rm
        self.age=age
        self.dis=dis
        self.rad=rad
        self.tax=tax
        self.ptratio=ptratio
        self.b=b
        self.lstat=lstat

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                'CRIM':[self.crim],
                'ZN':[self.zn],
                'INDUS':[self.indus],
                'CHAS':[self.chas],
                'NOX':[self.nox],
                'RM':[self.rm],
                'AGE':[self.age],
                'DIS':[self.dis],
                'RAD':[self.rad],
                'TAX':[self.tax],
                'PTRATIO':[self.ptratio],
                'B':[self.b],
                'LSTAT':[self.lstat],
            }
            return pd.DataFrame(custom_data_input_dict,index=[0])
        
        except Exception as e:
            raise CustomException(e,sys)