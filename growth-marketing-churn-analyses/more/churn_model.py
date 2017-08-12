from __future__ import division
import pandas as pd
import numpy as np

import json

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

from yhat import Yhat,YhatModel,preprocess

print "Importing data"
churn_df = pd.read_csv('data/churn.csv')

print "Formatting feature space"
# Isolate target data
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)

# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

print "Scaling features"
# This is important
scaler = StandardScaler()
X = scaler.fit_transform(X)

print "Generating training data"
train_index,test_index = train_test_split(churn_df.index)

# Write test data to file
test_churn_df = churn_df.ix[test_index]
test_churn_df.to_csv("test_churn.csv")

print "Training classifier"
clf = SVC(probability=True)
clf.fit(X[train_index],y[train_index])


class ChurnModel(YhatModel):
    # Type casts incoming data as a dataframe
    @preprocess(in_type=pd.DataFrame,out_type=pd.DataFrame)
    def execute(self,data):
        # Collect customer meta data
        response = data[['Area Code','Phone']]
        charges = ['Day Charge','Eve Charge','Night Charge','Intl Charge']
        response['customer_worth'] = data[charges].sum(axis=1)
        # Convert yes no columns to bool
        data[yes_no_cols] = data[yes_no_cols] == 'yes'
        # Create feature space
        X = data[features].as_matrix().astype(float)
        X = scaler.transform(X)
        # Make prediction
        churn_prob = clf.predict_proba(X)
        response['churn_prob'] = churn_prob[:,1]
        # Calculate expected loss by churn
        response['expected_loss'] = response['churn_prob'] * response['customer_worth']
        response = response.sort('expected_loss',ascending=False)
        # Return response DataFrame
        return response

yh = Yhat(
    "e[at]yhathq.com", 
    " MY APIKEY ", 
    "http://cloud.yhathq.com/" 
)

print "Deploying model"
response = yh.deploy("PythonChurnModel",ChurnModel,globals())

print json.dumps(response,indent=2)
