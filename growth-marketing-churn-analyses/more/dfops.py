from util import query_to_df,vectorize
from churndata import *
from sqlalchemy import *
import numpy as np
from datetime import datetime,timedelta
from sklearn.linear_model import LogisticRegression
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from util import occurred_in_last_k_days

db = create_engine('sqlite:///forjar.db')


metadata = MetaData(db)

Session = sessionmaker(bind=db)


session = Session()


def most_recent_visits():
     """
     Returns the most recent visits for a user, primarily used in featurizing
     """
     visits = session.query(Users).join(Visit).add_entity(Visit).group_by(Users.id).order_by(Visit.date.desc())
     return query_to_df(session, visits)



def most_recent_actions():
     """
     Returns the most recent events for a user, primarily used in featurizing
     """
     events = session.query(Users).join(Event).add_entity(Event).group_by(Users.id).order_by(Event.date.desc()).subquery()

     return query_to_df(session, events)



def user_activity_in_last_k_days(threshold):
    now = datetime.utcnow()
    days = now - timedelta(days=threshold)
    """
    Only grab the logins that occurred in the last 90 days
    """


    most_recent_user_visits = session.query(Users.id,Visit.date,Users.Campaign_ID).join(Visit,Users.id == Visit.user_id).order_by(Visit.date.desc()).group_by(Users.id).filter(Users.Campaign_ID == 'TW')

    df = query_to_df(session,most_recent_user_visits)
    df = df.drop('Users_Campaign_ID',axis=1)
    df['churned'] = df['visit_date'].apply(lambda x : x < days)
    df = df.reset_index()
    print type(df)
    return df

def user_visited_in_last_k_days(threshold):
    now = datetime.utcnow()
    days = now - timedelta(days=threshold)
    """
    Only grab the logins that occurred in the last 90 days
    """
    most_recent_user_visits = session.query(Users.id,Visit.date,Users.Campaign_ID).join(Visit,Users.id == Visit.user_id).order_by(Visit.date.desc()).group_by(Users.id).filter(Users.Campaign_ID == 'TW')

    df = query_to_df(session,most_recent_user_visits)
    df = df.drop('Users_Campaign_ID',axis=1)
    df['churned'] = df['visit_date'].apply(lambda x : x < days)
    df = df.reset_index()
    print type(df)
    return df


#df =  user_visited_in_last_k_days(90)

q = session.query(Users).join(Event).add_entity(Event)

df = query_to_df(session,q)



event_types = ['like','bought','share']


def is_last_90_days(group):
    return occurred_in_last_k_days(group,90)

def return_self(group):
    return group

def return_id(group):
    return group[0]


df['Event_date'] =  df.Event_date.apply(is_last_90_days)
df = df.drop(['Event_id','Event_User_Id','Users_Created_Date','Event_Meal_Id'],1)
df = df.groupby(['Users_id'])['Event_date'].aggregate(np.sum).reset_index()
print df
#groups = df.groupby(['Users_id','Event_Type','Event_date']).unique()
#print groups
#print groups.sort('Users_id')
#vectorized = vectorize(df,'churned')
