from __future__ import division
import matplotlib.pyplot as plt
from sqlalchemy import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from churndata import *
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame,Series
from pandas.core.groupby import GroupBy
from util import query_to_df
from util import *
db = create_engine('sqlite:///campaign-1.db')


metadata = MetaData(db)

Session = sessionmaker(bind=db)


session = Session()



"""
We only want events and users such that the user bought an item.
We count bought as $1 of revenue for simplicity.
"""

q = session.query(Users.Campaign_ID,Meal.Type,Event.Type).limit(300)

"""
Print out the counts by name.
This is a way of showing how to aggregate by campaign ids.
"""
df = query_to_df(session,q)

print df

transform_column(df,'Event_Type',event_to_num.get)
transform_column(df,'Users_Campaign_ID',campaign_to_num.get)
transform_column(df,'Meal_Type',meal_to_num.get)
print df
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








