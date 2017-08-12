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


"""
Counts the users by campaign id
"""
user_dist = session.query(Users)
user_df = query_to_df(session,user_dist)
transform_column(user_df,'Users_Campaign_ID',campaign_to_num.get)

hist_and_show(user_df,'Users_Campaign_ID')



q = session.query(Users.Campaign_ID,Event.Type,Users.id,Event.User_Id).filter(Event.Type == 'bought')
d = query_to_df(session,q)
print d.columns

transform_column(d,'Users_Campaign_ID',campaign_to_num.get)
"""
Show the counts for the event types
"""
transform_column(d,'Event_Type',event_to_num.get)
hist_and_show(d,'Users_Campaign_ID')







