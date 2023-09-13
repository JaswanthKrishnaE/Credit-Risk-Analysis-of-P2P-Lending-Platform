import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle
import csv
import os
import glob

from flask import Flask, render_template, request


df = pd.read_csv('LoanExport.csv',low_memory=False)


def feat_eng(data):
    try:
        data.set_index('LoanSeqNum', inplace=True)
    except:
        pass
    
    #Credit Score
    thresholds = [0, 650, 700, 750, 900]
    labels = ["Poor", "Fair", "Good", "Excellent"]
    data["NewCreditScore"] = pd.cut(data["CreditScore"], bins=thresholds, labels=labels, right=False)

    #LTV
    thresholds = [0,1,60,80,900]
    labels = ["High LTV", "Low LTV", "Moderate LTV", "High LTV"]
    data["LTV Group"] = pd.cut(data['LTV'], bins=thresholds, labels=labels, right = False, ordered=False)

    #Repayment Months
    thresholds = [0,48,96,144,192,240]
    labels = ["Quart1", "Quart2", "Quart3", "Quart4", "Quart5"]
    data["MonthsInRepayment Group"] = pd.cut(data['MonthsInRepayment'], bins=thresholds, labels=labels, right = False)
    
    # Feature Engineering 
    data["FirstPaymentDate"] = pd.to_datetime(data["FirstPaymentDate"], format="%Y%m")
    data["MaturityDate"] = pd.to_datetime(data["MaturityDate"], format="%Y%m")
    data['LoanDuration_days'] = (data['MaturityDate'] - data['FirstPaymentDate']).dt.days
    
    data['FirstPaymentYear'] = data['FirstPaymentDate'].dt.year
    data['FirstPaymentMonth'] = data['FirstPaymentDate'].dt.month
    data['MaturityYear'] = data['MaturityDate'].dt.year
    data['MaturityMonth'] = data['MaturityDate'].dt.month
    
    data["SellerName"] = data["SellerName"].fillna("Unknown User")
    data["NumBorrowers"] = data["NumBorrowers"].replace("X ", 99).astype(int)
    data['CreditScore_DTI_Combined'] = data['CreditScore'] * data['DTI']
    
    le = ['Occupancy', 'PostalCode','MSA', 'LoanPurpose', 'SellerName','ServicerName','PropertyState',
      'FirstTimeHomebuyer', 'NewCreditScore','LTV Group','MonthsInRepayment Group']
    data[le] = data[le].astype(str)
    le_ = LabelEncoder()
    for i in le:
        data[i] = le_.fit_transform(data[i])
        
    data = pd.get_dummies(data)
    
    return data

# Function to drop columns
def drop_cols(data):
    drop_col = ['ProductType','Units','Occupancy','OrigInterestRate','OrigLoanTerm','LoanDuration','FirstPaymentDate','MaturityDate']
    for i in drop_col:
        try:
            data = data.drop(i, axis=1)
        except:
            pass
    return data

final_pipeline = Pipeline([
    ('feat_eng',FunctionTransformer(feat_eng)),
    ('drop_cols',FunctionTransformer(drop_cols)),
    ('gnb',GaussianNB())
])

X = df.drop('EverDelinquent', axis=1)
y = df['EverDelinquent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

final_pipeline.fit(X_train,y_train)

y_predict = final_pipeline.predict(X_test)

pickle.dump(final_pipeline,open("classPipeline.pkl","wb"))

app = Flask(_name_)

# Load the classification pipeline from the pickle file
with open('classPipeline.pkl', 'rb') as f:
    classification_pipeline = pickle.load(f)

@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods = ['GET','POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        data = []
        
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
        data=pd.DataFrame(data)
        
        prediction = classification_pipeline.predict(data)
        return render_template('output.html',prediction=prediction)
        
if _name_ == '_main_':
    app.run(debug = True)