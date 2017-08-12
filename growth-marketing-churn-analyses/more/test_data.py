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
from pandas import DataFrame
from pandas.core.groupby import GroupBy
from util import query_to_df
import pandas as pd


from util import campaign_to_num,event_to_num,transform_column,hist_and_show,vectorize,to_percentage,num_rows,vectorize_label,meal_to_num,to_milliseconds
db = create_engine('sqlite:///forjar.db')


metadata = MetaData(db)

Session = sessionmaker(bind=db)


session = Session()

q = session.query(Event).join(Meal,Event.Meal_Id == Meal.id).join(Users).add_entity(Meal).add_entity(Users).filter(Event.Type == 'bought')

df = query_to_df(session,q)


def group_agg(group):
    return group

"""
Most of life time value is here.

We need to calculate the number of months each user stays
"""
df_user = df[['Meal_price','Event_date','Users_id']].groupby(['Users_id',df.Event_date.map(lambda x: (x.year,x.month)),df.Users_date.map(lambda x: (x.year,x.month))]).aggregate(np.mean)


print df_user.reset_index()

#df['Event_date'] = pd.to_datetime(df['Event_date'])

#print df.Users_id.groupby(df.Event_date.map(lambda x: (x.year,x.month))).value_counts()





