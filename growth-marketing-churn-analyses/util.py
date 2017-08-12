import matplotlib.pyplot as plt
from sqlalchemy import *
import numpy as np
from sqlalchemy.orm import sessionmaker
from churndata import *
from datetime import datetime,timedelta
from pandas import DataFrame
from time import mktime
from sklearn.feature_extraction import DictVectorizer


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



campaign_to_cost = {
    'TW' : .25,
    'RE' : .35,
    'FB' : .45,
    'PI' : .55
}



def occurred_in_last_k_days(datetime,k):
     now = datetime.now()
     days = now - timedelta(days=k)
     return datetime >= days


def expected_value(cf,campaign,avg_price):
    """
    Calculates the expected value of a given confusion matrix.
    Params:

    cf is a 2 x 2 confusion matrix
    campaign is a string representing the campaign from which the user came from
    avg_price is the avg monthly revenue made off of the user
    """
    cost_benefit = np.array([[avg_price,0],
                             [avg_price * campaign_to_cost[campaign],0]
    ])

    return np.dot(cf,cost_benefit)


def num_rows(df):
    return len(df.index)

def transform_column(df,column_name,fn):
    """
    Transforms a column with the given function
    """
    df[column_name] = df[column_name].apply(fn).astype('float')


def hist_and_show(df,column_name):
    """
    Histogram the given column for the data frame
    and render it
    """
    df.hist(column_name)
    plt.show()


def to_percentage(df,column_name):
    """
     Given a numeric field column name, converts each field
     to a percentage that a given element in a row contributed to the overall sum of a column
    """
    column_sum = df[column_name].sum()
    df[column_name] = df[column_name] / column_sum



def df_column_wise_norm(df):
     """
     Column wise norm. Calculates the norm of each column.
     The formula is:
           df - df.mean / df.max - df.min
     """
     df_norm = (df - df.mean()) / (df.max() - df.min())
     return df_norm


def query_to_df(session,query):
    """
    Convert an sql query to a pandas data frame
    """
    result = session.execute(query)
    d = DataFrame(result.fetchall())
    d.columns = result.keys()
    return d


def num_days_apart(first,second):
     return first - second.days



def sort_by_column(df,column):
    """
    This will sort the data frame by the given column with the maximum element coming first.
    """
    df = df.sort([column], ascending=False)


def to_milliseconds(dt):
    """
    Converts the given date time to epoch milliseconds
    """
    sec_since_epoch = mktime(dt.timetuple()) + dt.microsecond/1000000.0
    millis_since_epoch = sec_since_epoch * 1000
    return millis_since_epoch

def vectorize(df,label_column):
    """
    Vectorize input features wrt a label column.
    """
    feature_names = []
    for feature_name in df.columns.values:
        if feature_name != label_column:
            if label_column not in feature_names:
                feature_names.append(label_column)
    inputs = df[feature_names].index
    return inputs

def vectorize_label(df,label_column,num_labels,target_outcome):
    """
    Vectorize input features wrt a label column.
    """
    inputs = df[label_column].apply(lambda x: x== target_outcome).values

    return inputs


def confusion_matrix_for_example(num_below,num_above,num_total):
    """
    Confusion matrix for an individual example
    Params:
    num_below, the number below in the ranking for this example
    num_above, the number above in the ranking for this example
    num_total, the total number of examples
    returns: the probability for the individual example wrt confusion matrices
    """
    return np.asarray([num_above,num_total - num_above],[num_below,num_total - num_below]) / num_total



def confusion_matrices_for_df(df,probability_column):
    """
    Ranks the given data frame's rows by the given probability column
    then returns a list of confusion matrices in a sliding window for each element
    starting at the max and working its way down
    Params:
       df - the dataframe
       probability_column - the name of the probabiliity column
    """
    sort_by_column(df,probability_column)
    confusion_matrices_ret = []
    rows = num_rows(df)
    count = rows
    for row in df.iterrows():
        num_above = rows - count
        num_below = num_above - rows
        confusion_matrices_ret.append(confusion_matrix_for_example(num_above,num_below,rows))
    return confusion_matrices_ret






