#!/usr/bin/env python
# coding: utf-8

# # Data Set and the Domain Problem 
# 
# This dataset was originally posted on Kaggle. **The key task is to predict whether a product/part will go on backorder.**
# 
# Product backorder may be the result of strong sales performance (e.g. the product is in such a high demand that production cannot keep up with sales). However, backorders can upset consumers, lead to canceled orders and decreased customer loyalty. Companies want to avoid backorders, but also avoid overstocking every product (leading to higher inventory costs).
# 
# This dataset has ~1.9 million observations of products/parts in an 8 week period. The source of the data is unreferenced.
# 
# * __Outcome__: whether the product went on backorder
# * __Predictors__: Current inventory, sales history, forecasted sales, recommended stocking amount, product risk flags etc. (22 predictors in total)
# 
# The features and the target variable of the dataset are as follows:
# 
# **Description**
# ~~~
# # Features: 
# sku - Random ID for the product
# national_inv - Current inventory level for the part
# lead_time - Transit time for product (if available)
# in_transit_qty - Amount of product in transit from source
# forecast_3_month - Forecast sales for the next 3 months
# forecast_6_month - Forecast sales for the next 6 months
# forecast_9_month - Forecast sales for the next 9 months
# sales_1_month - Sales quantity for the prior 1 month time period
# sales_3_month - Sales quantity for the prior 3 month time period
# sales_6_month - Sales quantity for the prior 6 month time period
# sales_9_month - Sales quantity for the prior 9 month time period
# min_bank - Minimum recommend amount to stock
# potential_issue - Source issue for part identified
# pieces_past_due - Parts overdue from source
# perf_6_month_avg - Source performance for prior 6 month period
# perf_12_month_avg - Source performance for prior 12 month period
# local_bo_qty - Amount of stock orders overdue
# deck_risk - Part risk flag
# oe_constraint - Part risk flag
# ppap_risk - Part risk flag
# stop_auto_buy - Part risk flag
# rev_stop - Part risk flag
# 
# # Target 
# went_on_backorder - Product actually went on backorder
# ~~~
# 
# Two data files for training and testing are accessible in the JupyterHub environment.
#  
# <span style='background:yellow'>**NOTE:** The training data file is 117MB. **Do NOT add any data files to your commits** (training, test, or created), you may blow-through the _push limit_.</span>  
# You can easily lock up a notebook with bad coding practices.  
# Please save you project early, and often, and use `git commit` to checkpoint your process.

# 
# 
# --- 
# ### Overview / Roadmap
# 
# **General steps**:
# * Part 1: Preprocessing
#   * Dataset carpentry & Exploratory Data Analysis
#     * Develop functions to perform the necessary steps, you will have to carpentry the Training and the Testing data.
#   * Generate a **smart sample** of the the data
# * Part 2: Training and Validation
#   * Create 3 alternative pipelines, each does:
#       * Anomaly detection
#       * Dimensionality reduction
#       * Classification
# * Part 3: Testing
#   * Train chosen model full training data
#   * Evaluate model against testing
#   * Write a summary of your processing and an analysis of the model performance
# 
# 
# 

# ---
# 
# # Part 1: Data Preprocessing
# 
# In this part, we preprocess the given training set. 
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import os, sys
import itertools
import numpy as np
import pandas as pd


# ## Load dataset
# 
# **Description**
# ~~~
# sku - Random ID for the product
# national_inv - Current inventory level for the part
# lead_time - Transit time for product (if available)
# in_transit_qty - Amount of product in transit from source
# forecast_3_month - Forecast sales for the next 3 months
# forecast_6_month - Forecast sales for the next 6 months
# forecast_9_month - Forecast sales for the next 9 months
# sales_1_month - Sales quantity for the prior 1 month time period
# sales_3_month - Sales quantity for the prior 3 month time period
# sales_6_month - Sales quantity for the prior 6 month time period
# sales_9_month - Sales quantity for the prior 9 month time period
# min_bank - Minimum recommend amount to stock
# potential_issue - Source issue for part identified
# pieces_past_due - Parts overdue from source
# perf_6_month_avg - Source performance for prior 6 month period
# perf_12_month_avg - Source performance for prior 12 month period
# local_bo_qty - Amount of stock orders overdue
# deck_risk - Part risk flag
# oe_constraint - Part risk flag
# ppap_risk - Part risk flag
# stop_auto_buy - Part risk flag
# rev_stop - Part risk flag
# went_on_backorder - Product actually went on backorder. 
# ~~~
# 
# **NOTE**: This is a real-world dataset without any preprocessing. There will also be warnings due to fact that the 1st column is mixing integer and string values. The last column, `went_on_backorder`, is what we are trying to predict.
# 

# In[2]:


# location of the training set; for Parts 1 and 2, 
# you only have access to this training data set for Parts 1 and 2 
DATASET = '/dsa/data/all_datasets/back_order/Kaggle_Training_Dataset_v2.csv'
assert os.path.exists(DATASET)

# Load and shuffle
dataset = pd.read_csv(DATASET).sample(frac = 1).reset_index(drop=True)

dataset.head().transpose()


# In[3]:


dataset.describe().transpose()


# ---
# 
# ## Processing
# 
# In this section, the goal is to figure out:
# 
# * which columns we can use directly,  
# * which columns are usable after some processing,  
# * and which columns are not processable or obviously irrelevant (like product id) that we will discard.
# 
# Then process and prepare this dataset for creating a predictive model.
# 
# **You can choose to employ any kind of approach for Exploratory Data Analysis to understand the data better. It is up to you to make the decisions regarding the cleaning/reencoding/dropping/imputing/binning etc.**  The following cells only serve as guidance. 

# In[4]:


dataset.info()


# ### Take samples and examine the dataset

# In[5]:


dataset.iloc[:3,:6]


# In[6]:


dataset.iloc[:3,6:12]


# In[7]:


dataset.iloc[:3,12:18]


# In[8]:


dataset.iloc[:3,18:24]


# In[9]:


dataset.isnull().sum()


# In[10]:


# Useful vars: national_inv, lead_time, in_transit_qty, sales(1,3,6,9), perf(6,12), local_bo_qty
# Maybe vars: min_bank
# Not useful: part risk flags??, sku


# In[11]:


# Drop rows in lead_time that are null (only 100,000), since product time might be affected by imputation


# ### Drop columns that are obviously irrelevant or not processable

# In[3]:


# Add code below this comment  (Question #E101)
# ----------------------------------
dataset.drop(columns = ["sku"], inplace = True)


# In[4]:


dataset = dataset.dropna(axis = 0)
dataset.isnull().sum()


# In[12]:


dataset.shape


# In[13]:


dataset.describe(include='all').loc['unique', :]


# In[ ]:


# All object types have 2 unique values, no missing or null values


# In[14]:


plt.scatter(data = dataset, x = "went_on_backorder", y = "lead_time")
plt.show()


# There seem to be some extreme outliers at around 50, considering most commonly 20 and less are put on back order.  It seems to be slightly more common to have longer times in the no category.

# In[15]:


sales_list = ['sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month']

for i in sales_list:
    plt.scatter(data = dataset, x = i, y = "went_on_backorder")
    plt.xlabel("{}".format(i))
    plt.ylabel("Went on Backorder")
    plt.show()


# The Number of sales does not seem to affect back order or not, as all the plots look the same for 1,3,6,9 months.

# It looks like most items do not have a minimum number of inventory

# In[18]:


import seaborn as sns
sns.heatmap(data = dataset.corr(), cmap = 'coolwarm')
plt.show()


# Going to do some feature engineering on the dataset.

# In[5]:


# Feature Engineering functions

def lead_time_cat(data):
    low_thres = np.percentile(data['lead_time'], 0.25)
    high_thres = np.percentile(data['lead_time'], 0.75)
    
    data['lead_time_cat'] = np.where(data['lead_time'] <= low_thres, 'low',
                                     np.where(data['lead_time'] <= high_thres, 'medium', 'high'))
    return data

def sales_forecast_cat(data):
    sales_columns = [col for col in data.columns if re.search('sales', col)]
    forecast_columns = [col for col in data.columns if re.search('forecast', col)]
    
    for col in sales_columns + forecast_columns:
        low_thres = np.percentile(data[col], 0.25)
        high_thres = np.percentile(data[col], 0.75)
        category_column = '{}_cat'.format(col)
        data[category_column] = np.where(data[col] <= low_thres, 'low',
                                         np.where(data[col] <= high_thres, 'medium', 'high'))
    
    return data

def min_bank_cat(data):
    threshold = np.percentile(data['min_bank'], 0.75)
    data['min_bank_cat'] = np.where(data['min_bank'] > threshold, 'low', 'high')
    return data


# In[6]:


import re
min_bank_cat(dataset)
sales_forecast_cat(dataset)
lead_time_cat(dataset)


# ### Find unique values of string columns
# 
# Now, try to make sure that the Yes/No columns really only contain Yes or No.  
# If that's true, proceed to convert them into binaries (0s and 1s).
# 
# **Tip**: use [unique()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.unique.html) function of pandas Series.
# 
# Example
# 
# ~~~python
# print('went_on_backorder', dataset['went_on_backorder'].unique())
# ~~~

# In[7]:


# All the column names of these yes/no columns
yes_no_columns = list(filter(lambda i: dataset[i].dtype!=np.float64, dataset.columns))
print(yes_no_columns)

# Add code below this comment  (Question #E102)
# ----------------------------------
from sklearn.preprocessing import LabelEncoder

def encoder(data):
    le = LabelEncoder()
    for i in data.columns:
        if data[i].dtype != np.float64:
            if len(data[i].unique()) > 2:
                data = pd.get_dummies(data, columns=[i])
            else:
                data[i] = le.fit_transform(data[i])
        else:
            continue
    return data
            
encoder(dataset)


# You may see **nan** also as possible values representing missing values in the dataset.
# 
# We fill them using most popular values, the [Mode](https://en.wikipedia.org/wiki/Mode_%28statistics%29) in Stats.

# In[ ]:


# dropped NA values, so no need to impute

for column_name in yes_no_columns:
    mode = dataset[column_name].apply(str).mode()[0]
    print('Filling missing values of {} with {}'.format(column_name, mode))
    dataset[column_name].fillna(mode, inplace=True)


# ### Convert yes/no columns into binary (0s and 1s)

# In[ ]:


# Add code below this comment  (Question #E103)
# ----------------------------------



    


# Now all columns should be either int64 or float64.

# In[10]:


dataset.info()


# ### Smartly sample the data into a more manageable size for cross-validation 
# 
# 
# This is a good point to re-balance dataset before actually moving on. For sampling, we can either take advantage of pandas/numpy `sample` method or use `imblearn` [package](https://imbalanced-learn.org/stable/user_guide.html#user-guide). 
# 
# 
# Create a "smart", balanced sample: 
# 
#  * it should be balanced: it is up to you if you do a 50/50 balance or anything else. 
#  * it should be manageable-sized: you will run a lot of cross validations in Part 2 that are very time consuming. 
#  * it should still reflect the characteristics of the original data.
#  
# Use any approach you deem necessary to create the balanced sample. It will serve as your data set in the development.  
# 
# 

# In[27]:


num_backorder = np.sum(dataset['went_on_backorder']==1)
print('backorder ratio:', num_backorder, '/', len(dataset), '=', num_backorder / len(dataset))


# In[8]:


# Spliting into X and y
X = dataset.drop(columns = ['went_on_backorder', 'lead_time_cat'])
y = dataset.went_on_backorder


# In[9]:


del dataset


# In[10]:


# Selected RandomUnderSampler due to memory contraints
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
#sampler = RandomUnderSampler(random_state=42)
#X_sampled, y_sampled = sampler.fit_resample(X,y)
#print(X_sampled.shape, y_sampled.shape)

def hybrid_sampling(X, y):
    rand_sam = RandomUnderSampler(sampling_strategy = 0.2,random_state=42)
    X_sampled, y_sampled = rand_sam.fit_resample(X, y)
    
    over_sam = RandomOverSampler(random_state=42)
    X_sampled, y_sampled = over_sam.fit_resample(X_sampled, y_sampled)
    
    return X_sampled, y_sampled


# In[13]:


from imblearn.under_sampling import RandomUnderSampler
rand_sam = RandomUnderSampler(sampling_strategy = 0.2,random_state=42)
X_sampled, y_sampled = rand_sam.fit_resample(X, y)
print(X_sampled.shape)
print(y_sampled.shape)


# In[12]:


from imblearn.over_sampling import RandomOverSampler
over_sam = RandomOverSampler(random_state=42)
X_sampled, y_sampled = over_sam.fit_resample(X_sampled, y_sampled)
print(X_sampled.shape)
print(y_sampled.shape)


# In[11]:


X_sampled, y_sampled = hybrid_sampling(X,y)
print(X_sampled.shape)
print(y_sampled.shape)


# In[40]:


rand_num_backorder = np.sum(y_sampled==1)
print('backorder ratio:', num_backorder, '/', len(X_sampled), '=', num_backorder / len(X_sampled))


# In[12]:


# looking at the descriptive stats to see how that data has changed
X_sampled.describe().T


# In[42]:


sns.heatmap(X_sampled.corr(), cmap = 'coolwarm')
plt.show()


# Looks like we lost some of the linear correlations between the sales and forecasting variables.

# Create a smart sample of the data and save it.  You can either store the data to csv files or simply use `joblib` to dump the variables and load them in Part 2. 
# 
# **Example code for using joblib:**
# 
# Say we need to store three objects (sampled_X, sampled_y, model) to a file. 
# 
# ```python
# import joblib
# 
# # for dumping 
# joblib.dump([sampled_X, sampled_y, model], 'data/sample-data-v1.pkl')
# 
# # for loading
# sampled_X, sampled_y, model = joblib.load('data/sample-data-v1.pkl')
# 
# ```
# 

# In[13]:


# Add code below this comment   (Question #E104) 
# ----------------------------------
import joblib

joblib.dump([X_sampled, y_sampled], 'sample-data.pkl')


# 
# **Note:** After sampling the data, you may want to write the data to a file for reloading later.
# 
# <span style="background: yellow;">If required, remove the `dataset` variable to avoid any memory-related issue.</span> 

# In[14]:


# Write your smart, balanced sample to local file  
# ----------------------------------
data_list = [X_sampled,pd.DataFrame(y_sampled)]
combined = pd.concat(data_list)

combined.to_csv("sample-data.csv")


# 
# **Make commits often and definitely when done!**  
# Comment should be: `Final Project, Checkpoint - Data Sampled`
# 

# # Save your notebook!
# ## Then commit and push 
