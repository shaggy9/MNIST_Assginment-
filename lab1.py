# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:19:09 2019

@author: user
"""

import pandas as pd
housing=pd.read_csv('C://Users//user//Desktop//housing.csv')
housing.head()
housing.info() 
housing["ocean_proximity"].value_counts() #count in a particular column(ocean_proximity) and show how many values from each category it includes
# Summary stats
housing.describe() # shows the descretive statistics (mean,std,min,etc)
# Plot histograms 
import matplotlib.pyplot as plt
# If you are running this using Jupyter notebook then also include the line %matplotlib inline
housing.hist(bins=50, figsize=(12,7))
plt.show()
#
# Create training and test sets
#

# Split into training and test sets, evalute the dataset
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head() # 
train_set.head()
# Stratified sampling illustration

# Looking at spread of median income
housing["median_income"].hist()
plt.show()
# Introduce a new column in the data frame...
# Divide by 1.5 to limit the number of income categories
import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].hist()
plt.show()
# Use StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# split generated two sets of indices based on preserving distribution of income_cat attribute 
# use these to create train and test sets
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_train_set.describe()
strat_test_set.describe()
# check histograms...
strat_train_set.hist(bins=50, figsize=(12,7))
strat_test_set.hist(bins=50, figsize=(12,7))
plt.show()
# Having created these can then remove the income_cat column
# One way
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
#
# Visualise the data
#
# Create a copy of the training set
housing = strat_train_set.copy()
# scatter pot
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
# alpha level controls transparency - opaqueness
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.show()


#
# Investigate correlations
#

corr_matrix = housing.corr()
corr_matrix
corr_matrix["median_house_value"].sort_values(ascending=False)
# plotting correlations
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,7))
plt.show()
# Look at pairs of attributes in detail...
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])

# Combining attributes...
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
# Look at modified data frame
housing.head(10)
# Can then look at correlations again...


#
# Preparing the data
#

# drop labels from the training set and create a separate data frame with the target variable
housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy()
# check for nulls
housing.isnull().any()

# Ways of dealing with missing values...
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

# Drop affected rows
sample_incomplete_rows.dropna(subset=["total_bedrooms"])  # option 1

# Drop the column
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2

# Replace with median
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows


# Go for option 3
housing["total_bedrooms"].fillna(median, inplace=True)

# And then recompute any dependencies and add in other attribute combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


#
# Categorical data - transforming and one-hot encoding
#

# Isolate categorical attribute
housing_cat = housing['ocean_proximity']
housing_cat.head(10)

# factorise to turn into integers (note different aproach from the book)
housing_cat_encoded, housing_categories = housing_cat.factorize()

housing_cat_encoded
len(housing_cat_encoded)
housing_categories

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
# Need to reshape into 2-d array
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot

# Sparse matrix - can convert to dense array
hcea = housing_cat_1hot.toarray()
# Can take a look at this
hcea[:10]


# Transforming numeric data
# Need to isolate numeric attributes
housing_num = housing.drop('ocean_proximity', axis=1)

cat_attribs = ['ocean_proximity']

num_attribs = list(housing_num)


# Perform scaling using StandardScaler
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
housing_num[num_attribs] = std_scaler.fit_transform(housing_num[num_attribs])

# Take a look at this...
housing_num.head(10)


# 
# Now add the encoded categorical data to the transformed numeric data
#
enc_data = pd.DataFrame(housing_cat_1hot.toarray())
enc_data.columns = housing_categories
enc_data.index = housing.index

housing_prepared = housing_num.join(enc_data)

# Take a look at this...
housing_prepared.head(10)
housing_prepared.describe()
housing_prepared.shape

#
# Select and train model....
#

# Try out linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Looking at just a subset of the data for illustrative purposes
some_data = housing_prepared.iloc[:5]
some_labels = housing_labels.iloc[:5]

print("Predictions:", lin_reg.predict(some_data))

print("Labels:", list(some_labels))


# Run on whole dataset and calculate rmse 
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# Alternatively using cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
