Extract transform load - vectorizing Data

Overview
========================

The core objective of this exercise is to understand sql queries, pandas data frames, and general exploratory data analysis.

Towards the end, we will learn how to take our basic tools for exploratory data analysis and transform the inputs in to something

appropriate for a machine learning algorithm.


For our baseline classes, we will be using the classes from churndata.py . This contains all of the necessary classes from which

we will derive all of our analysis.

The associated database forjar.db contains our actual data set. We will need to use sql alchemy to load the data in.




Loading Data Via SQLAlchemy
====================================

First we will be loading the data from our sql lite database and doing a simple join.

Our objective here will be to get a list of the users who have bought something.

If we think in terms of our objectives of the site, it is to maximize revenue.

In any given day, we need to understand which of our users actually tend to buy things on the site.

If they aren't buying anything, we should do something about it given the data we know about them.


Exercise:

     1.  Goal: Load data from an sql database such that each you load a join of the users and events. We only want the users who bought something.

         Steps:
              1. Load data via sqlalchemy from the sql lite database forjar.db
              2. Create a query that contains a join on events and a filter on bought. Look in to sqlalchemy sessions.


     2. Load the results of a call to query.all() in to a data frame.

         Steps:
              1. Create a dataframe with pandas
              2. Set the columns to the query keys

The goal with this particular exercise is to understand which users are buying things so we can understand what attributes are successful.

Resources:
SQLLite/SQLAlchemy - http://docs.sqlalchemy.org/en/rel_0_9/dialects/sqlite.html
Basic Querying -     http://docs.sqlalchemy.org/en/rel_0_9/orm/query.html
Pandas Data Structures - http://pandas.pydata.org/pandas-docs/stable/dsintro.html


If you are still stuck, here is a solution:

https://gist.github.com/agibsonccc/c5f34f6a5d24e041d535


Ranking
======================

Now we will want to perform some sort of ranking, understanding attributes of who bought the most will allow us

to understand who our most profitable users are. Users who buy the most do not necessarily have the highest life time value,

but it is a great low hanging fruit for understanding where to begin understanding your users.


Exercise:
         Create a ranked grouping of the users who bought the most

         Steps:
                1. Using our previous query, we should be able to also rank the users who bought the most.
                2.  Again put the data in to a dataframe. RUn your aggregations and group bys via the data frame.

Resources:

http://docs.sqlalchemy.org/en/rel_0_9/orm/query.html



Solution:
https://gist.github.com/agibsonccc/7798e0908ab975a5f376


Counts on campaign wise event actions
========================================


Now we will use Pandas to start doing some exploratory analysis.  Let's compute some fairly simple statistics on our data.

Exercise:

     Steps:
           1. Load the data from sql alchemy in to a pandas data frame

           2. Using pandas, calculate the mean number of times each user

           3. Do a dual group by from each campaign type (facebook,twitter,pinterest,...) and each event type



Resources:

http://wesmckinney.com/blog/?p=125

Solution:
https://gist.github.com/agibsonccc/f5538976ed3782cb0441



Visualizing Data
============================

Pandas has very powerful plotting built in to it alongside matplotlib. Let's generate scatterplots for all of the various user campaign to event types.

Pandas has a lot of built in tools for data vis. Underneath it uses matplot lib. One key thing of note here is that pandas will not actually render your plots for you.

To render plots after a call to something like dataframe.hist(), do the following:

     import matplotlib.pyplot as plt
     plt.show()

This will allow us to see correlations in events all at once.

Steps:
       1. Load the data and do a join on: (Users.Campaign_ID,Event.Type,Users.id,Event.User_Id
       2. If you have more than these columns in your dataframe subset them to this list of columns
       3. Run an ordinal transform on each column. This is done via a 1 of k encoding mentioned earlier. (Look in to the dict.get function alongside df[columnname].apply()
       4. Now iterate over every possible combination of columns and plot a scatter plot of each. Render these on the screen all at once.
          Look in to plt.subplot for this. The end goal here is to transform the string values in each column in the data frame to a numerical representation.



Machine Learning Data input prep
===========================================

For Machine Learning Algorithms, they can only accept numbers. Our specific task here will be to predict a label.


Exercise:

Let's build out a data frame such that we have an outcome label. Set the outcome label to be event type.

From here, binarize the event type outcome to be == bought or not.

Solution:
https://gist.github.com/agibsonccc/9c54fbdc8d6f9b3f53fb


Data Normalization
=====================================================

Machine Learning algorithms typically work better if you scale the data (squish the data in to 0,1 range)

Exercise:


       Query for Users.Campaign_ID,Meal.Type,Event.Type and load it in to a dataframe

       Transform the data in to numerical (ordinal etc, think about what we did before)

       Split out the feature set columns from the outcome label and normalize the given features.

Resources:

http://scikit-learn.org/stable/modules/preprocessing.html

Solution:
https://gist.github.com/agibsonccc/ec5062e81a4817cf35d4
