import matplotlib.pyplot as plt
from sqlalchemy import *
import numpy as np
from sqlalchemy.orm import sessionmaker
from churndata import *
from pandas import DataFrame
from util import query_to_df
from util import campaign_to_num,event_to_num,transform_column,hist_and_show,vectorize
db = create_engine('sqlite:///forjar.db')


metadata = MetaData(db)

Session = sessionmaker(bind=db)


session = Session()


def transform_column(df,column_name,fn):
    """
    Transforms a column with the given function
    """
    df[column_name] = df[column_name].apply(fn).astype('float')

campaign_to_num = {
	'TW' : 1,
	'RE' : 2,
    'FB' : 3,
    'PI' : 4
}

event_to_num = {
   'like' : 1,
   'share' : 2,
   'nothing' : 3,
   'bought' : 4
}



meal_to_num = {
   'japanese':  1,
   'chinese' : 2,
   'french' : 3,
    'german' : 4,
    'italian' : 5,
    'mexican' : 6,
    'vietnamese' : 7
}


"""
Counts the users by campaign id
"""
user_dist = session.query(Users)
user_df = query_to_df(session,user_dist)
transform_column(user_df,'Users_Campaign_ID',campaign_to_num.get)




q = session.query(Users.Campaign_ID,Event.Type,Users.id)
d = query_to_df(session,q)

column_transforms = {
    'Users_Campaign_ID' : campaign_to_num.get,
    'Event_Type' : event_to_num.get
}

sub_plot_size = len(d.columns) * len(d.columns)
"""
Subplot call here
"""
for column in d.columns:
    if column_transforms.has_key(column):
        print 'Transforming ' + column
        transform_column(d,column,column_transforms[column])


count = 1
fig = plt.figure()
for column in d.columns:
    for column2 in d.columns:
        x = d[column]
        y = d[column2]
        print (x,y)
        print('Plotting ',column,column2)
        fig.add_subplot(1,sub_plot_size,count)
        count = count + 1
        plt.scatter(x,y)

plt.show()
