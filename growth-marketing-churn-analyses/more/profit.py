from __future__ import division
import matplotlib.pyplot as plt
from sqlalchemy import *
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from churndata import *
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame,Series
from pandas.core.groupby import GroupBy
import pandas as pd
from util import query_to_df
from util import *
import matplotlib.pyplot as plt
db = create_engine('sqlite:///campaign-1.db')


metadata = MetaData(db)

Session = sessionmaker(bind=db)


session = Session()
"""
We only want events and users such that the user bought an item.
We count bought as $1 of revenue for simplicity.
"""

q = session.query(Users.Campaign_ID,Event.Type).filter(Event.Type == 'bought')

"""
Print out the counts by name.
This is a way of showing how to aggregate by campaign ids.
"""
df = query_to_df(session,q)
"""
Basic statistics collecting; calculate the sum for each row.
"""
sum = 0
for campaign_id in campaign_to_num.keys():
    rows = num_rows(df.groupby('Users_Campaign_ID').get_group(campaign_id))
    sum = sum + rows
    print 'id was ' + campaign_id + ' ' + str(rows)

"""
Print the percentage of buy actions for each campaign id
"""
for campaign_id in campaign_to_num.keys():
    rows = num_rows(df.groupby('Users_Campaign_ID').get_group(campaign_id))
    rows = rows / sum
    print 'id was ' + campaign_id + ' and percent of customers bought was ' + str(rows)



"""
Calculate the lift curves

We will take our first data set and use it as a baseline.

We will assume that each user gained was the result of a marketing campaign on each social network.

Our goal here is to calculate the lift curve such that when we run a second campaign, we can see

either improvements or not with respect to the delta. The positive response rate will be marked

by the number of buy actions within the dataset, from there we can calculate a conversion rate.

After wards, we will run similar calculations to visualize a lift curve.

"""



"""
We only want events and users such that the user bought an item.
We count bought as $1 of revenue for simplicity.
"""

churn_df = pd.read_csv('data/churn.csv')
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)
to_drop = ['State','Area Code','Phone','Churn?']
#to_drop = ['has_event_mo_6']
churn_feat_space = churn_df.drop(to_drop,axis=1)
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
features = churn_feat_space.columns
#X = churn_feat_space.fillna(0).as_matrix().astype(np.float)
X = churn_feat_space.as_matrix().astype(np.float)

print "Scaling features"
# This is important
scaler = StandardScaler()
X = scaler.fit_transform(X)

train_index,test_index = train_test_split(churn_df.index)



"""
"""


q = session.query(Users.Campaign_ID,Meal.Type,Event.Type).limit(300)

"""
Print out the counts by name.
This is a way of showing how to aggregate by campaign ids.
"""
df = query_to_df(session,q)


transform_column(df,'Event_Type',event_to_num.get)
transform_column(df,'Users_Campaign_ID',campaign_to_num.get)
transform_column(df,'Meal_Type',meal_to_num.get)

"""
Prediction scores.

"""
data_set = vectorize(df,'Event_Type')
labels =  vectorize_label(df,'Event_Type',2,4)


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(data_set, labels, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


y_test = y[test_index].astype(float)

# Cost-Benefit Matrix
cb = np.array([[4, -5],
               [0, 0]])

# Define classifiers for comparison
classifiers = [("Random Forest", RF),
               ("Logistic Regression", LR),
               ("Gradient Boosting Classifier", GBC)]
               

def confusion_rates(cm): 

    tn = cm[0][0]
    fp = cm[0][1] 
    fn = cm[1][0]
    tp = cm[1][1]
    
    N = fp + tn
    P = tp + fn
    
    tpr = tp / P
    fpr = fp / P
    fnr = fn / N
    tnr = tn / N
    
    rates = np.array([[tpr, fpr], [fnr, tnr]])
    
    return rates


def profit_curve(classifiers):
    for clf_class in classifiers:
        name, clf_class = clf_class[0], clf_class[1]
        clf = clf_class()
        fit = clf.fit(X[train_index], y[train_index])
        probabilities = np.array([prob[0] for prob in fit.predict_proba(X[test_index])])
        profit = []
        
        indicies = np.argsort(probabilities)[::1]
    
        for idx in xrange(len(indicies)): 
            pred_true = indicies[:idx]
            ctr = np.arange(indicies.shape[0])
            masked_prediction = np.in1d(ctr, pred_true)
            cm = confusion_matrix(y_test.astype(int), masked_prediction.astype(int))
            rates = confusion_rates(cm)
     
            profit.append(np.sum(np.multiply(rates,cb)))
        
        plt.plot((np.arange(len(indicies)) / len(indicies) * 100), profit, label=name)
    plt.legend(loc="lower right")
    plt.title("Profits of classifiers")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.show()

# Plot profit curves
profit_curve(classifiers)











