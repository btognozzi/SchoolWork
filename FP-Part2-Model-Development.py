#!/usr/bin/env python
# coding: utf-8

# # Part 2: Model Development
# 
# In this part, we develop three unique pipelines for predicting backorder. We **use the smart sample from Part 1** to fit and evaluate these pipelines. 
# 
# 
# **By unique, we mean that if an ML method (i.e. classification,  feature selection, or anomaly detection) is used in Pipeline 1, that same method should not be used in Pipeline 2 or Pipeline 3, etc.** 
# 
# You are **free to pick any models** from Scikit-Learn or any custom models that work within sklearn pipeline. Below is a pool of methods that you can choose from.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import os, sys
import itertools
import numpy as np
import pandas as pd


# ## Reload the smart sample here

# In[2]:



# Reload your smart sample from local file 
# ----------------------------------
import joblib

X,y = joblib.load('sample-data.pkl')


# ## Split the data into Train/Test
# 
# Note that `train_test_split` does NOT do stratified sampling by default. If you have decided to have some imbalance in your sample, make sure to use the `stratify` option. 

# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ## Developing Pipeline
# 
# In this section, we design an operationalized machine learning pipeline, which includes:
# 
# * Anomaly detection
# * Scaling/normalization (optional) 
# * Dimensionality Reduction
# * Train a classification model
# 
# 
# You are free to use any of the models that we learned in the past or you cab use new models. Here is a pool of methods you can choose from: 
# 
# ### Pool of Anomaly Detection Methods 
# 1. IsolationForest
# 2. EllipticEnvelope
# 3. LocalOutlierFactor
# 4. OneClassSVM
# 5. SGDOneClassSVM
# 
# ### Pool of Feature Selection Methods 
# 
# 1. VarianceThreshold
# 1. SelectKBest with any scoring method (e.g, chi, f_classif, mutual_info_classif)
# 1. SelectKPercentile
# 3. SelectFpr, SelectFdr, or  SelectFwe
# 1. GenericUnivariateSelect
# 2. PCA
# 3. Factor Analysis
# 4. Variance Threshold
# 5. RFE
# 7. SelectFromModel
# 
# 
# ### Classification Methods 
# 1. Decision Tree
# 2. Random Forest
# 3. Logistic Regression
# 4. Naive Bayes
# 5. Linear SVC
# 6. SVC with kernels
# 7. KNeighborsClassifier
# 8. GradientBoostingClassifier
# 9. XGBClassifier
# 10. LGBM Classifier
# 
# ---
# 
# It is difficult to fit an anomaly detection method in the sklearn pipeline without writing custom codes. For simplicity, we can avoid fitting an anomaly detection method within the pipeline. So, we can create the workflow in two steps:
# * Step I: fit an outlier with the training set
# * Step II: define a pipeline using a feature selection and a classification method. Then cross-validate this pipeline using the training data without outliers. 
# 
# 
# **Optional:** Those who are interested in writing custom codes for adding an outlier detection method into the sklearn pipeline, please consult this [notebook](../notes/Outlier_Rejection_Pipeline.ipynb).
# 
# 
# <span style='background:yellow'>We will be using the search strategies we have learned to find the best combination of hyperparameters of the pipelines.</span>
# 
# * Note: if your smart sample is somewhat imbalanced, you might want to change the scoring method in GridSearchCV (see the [doc](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)) or RandomizedSearchCV ( see the [doc](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)).
# 
# Once we tune the pipeline hyperparameters, we identify the best model and give an evaluation **using the test set that we created here in Part II**. For evaluation, you report confusion matrix, precision, recall, f1-score, accuracy, and other measures if you like. 
# 
# 
# You can add more notebook cells or import any Python modules as needed.

# In[4]:


# 1st pipeline packages and functions
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# ---
# 
# 
# ### Your 1st pipeline 
#   * Anomaly detection
#   * Dimensionality reduction
#   * Model training/validation
#   
# Add cells as needed. 

# In[5]:


# Add anomaly detection code  (Question #E201)
# ----------------------------------
def iso_forest(X,y,contamination='auto'):
    forest = IsolationForest(max_samples=100, contamination=contamination, random_state=42)
    forest.fit(X)
    y_pred = forest.predict(X)
    return X[y_pred == 1], y[y_pred == 1]


# In[6]:


iso_X_train, iso_y_train = iso_forest(X_train,y_train)
iso_X_test, iso_y_test = iso_forest(X_test, y_test)


# In[7]:


# Add code for pipeline with feature selection and classification and hyperparameter tuning  (Question #E202)
# ----------------------------------
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA()),
    ('xgb', XGBClassifier())
])


# In[8]:


param_grid = {
    'pca__n_components':np.linspace(10,20,num=3,dtype=int),
    'xgb__n_estimators':np.linspace(100,300,num=3,dtype=int),
    'xgb__learning_rate':np.logspace(-2,0),
    'xgb__subsample':np.linspace(0.5,1,num=6),
    'xgb__tree_method':['hist'],
    'xgb__max_bins':np.logspace(1,3),
    'xgb__use_label_encoder':[False],
    'xgb__eval_metric':['logloss'],
    'xgb__objective':['binary:logistic'],
    'xgb__random_state':[42]
}


# In[9]:


from sklearn.model_selection import RandomizedSearchCV

grid_model = RandomizedSearchCV(pipe, param_distributions = param_grid, n_iter=20, random_state=42)


# In[10]:


grid_model.fit(iso_X_train, iso_y_train)
print(grid_model.best_estimator_)


# In[11]:


# Give an evaluation  (Question #E203)
# ----------------------------------
from sklearn.metrics import classification_report, confusion_matrix
# show tuning results: display scores, best model's parameters, classification report and confusion matrix for TRAINING data
print(grid_model.best_score_)

xgb_pred = grid_model.predict(iso_X_train)

print(classification_report(y_true = iso_y_train, y_pred = xgb_pred))
print(confusion_matrix(y_true = iso_y_train, y_pred = xgb_pred))


# #### <center>Record the best hyperparameters and performance resulting from this pipeline.</center>
# Detail Hyperparameters and Results below  (Question #E204)
# ---------------------------------------------

This model performed well on the training data with 94% CV accuracy.  Precision, Recall, and F1 all are 99% or 98%.

Model has performed very well with the parameter tuning done.

PCA: 15 components

Best parameters are as follows:
    eval_metric: 'logloss'
    importance_type: 'gain'
    learning_rate: 0.83
    max_bins(used for the 'hist' tree method): 19.3
    max_depth: 6
    n_estimators: 300
    subsample: 0.5
    tree_method(for faster training, relatively the same accuracy than using the default 'exact'): 'hist'


Full details below:
XGBClassifier(base_score=0.5, booster='gbtree',
                               colsample_bylevel=1, colsample_bynode=1,
                               colsample_bytree=1, eval_metric='logloss',
                               gamma=0, gpu_id=-1, importance_type='gain',
                               interaction_constraints='',
                               learning_rate=0.8286427728546842,
                               max_bins=19.306977288832496, max_delta_step=0,
                               max_depth=6, min_child_weight=1, missing=nan,
                               monotone_constraints='()', n_estimators=300,
                               n_jobs=18, num_parallel_tree=1, random_state=42,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                               subsample=0.5, tree_method='hist',
                               use_label_encoder=False, validate_parameters=1,
                               verbosity=None)
# ## <span style="background: yellow;">Commit your code!</span> 

# ---
# 
# 
# ### Your 2nd pipeline
#   * Anomaly detection
#   * Dimensionality reduction
#   * Model training/validation

# In[4]:


# Packages and libraries for 2nd pipline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[5]:


# Add anomaly detection code  (Question #E205)
# ----------------------------------
def lof(X,y,neighbors):
    lof = LocalOutlierFactor(contamination = 'auto', n_neighbors = neighbors)
    y_pred = lof.fit_predict(X)
    return X[y_pred == 1], y[y_pred == 1]


# In[6]:


neighbors_list = [10,20,30]

for i in neighbors_list:
    X_new, y_new = lof(X_train, y_train, i)
    print("Shape of Dataset using {} Neighbors:".format(i), X_new.shape, y_new.shape)


# In[6]:


# I will go with 10 Neighbors
lof_X_train, lof_y_train = lof(X_train, y_train, 10)


# In[7]:


# Add code for pipeline with feature selection and classification and hyperparameter tuning  (Question #E206)
# ----------------------------------
pipe2 = Pipeline([
    ('var_thresh', VarianceThreshold()),
    ('scale', StandardScaler()),
    ('select', SelectKBest()),
    ('rfc', RandomForestClassifier())
])


# In[8]:


param_grid_2 = {
    'var_thresh__threshold': [0, 0.1, 0.2],
    'select__k':np.linspace(5,15,num=6, dtype=int),
    'rfc__n_estimators':np.linspace(100,300,num=3,dtype=int),
    'rfc__criterion':['gini'],
    'rfc__max_features':['sqrt', 'log2'],
    'rfc__min_samples_split': np.linspace(2,5,num=4,dtype=int),
    'rfc__min_samples_leaf':np.linspace(1,3,num=3,dtype=int),
    'rfc__random_state':[42]
}


# In[13]:


from sklearn.model_selection import RandomizedSearchCV

grid_model2 = RandomizedSearchCV(pipe2, param_distributions = param_grid_2, n_iter=20, n_jobs = 5, random_state=42)


# In[14]:


grid_model2.fit(lof_X_train, lof_y_train)
print(grid_model2.best_estimator_)


# In[15]:


# Give an evaluation  (Question #E207)
# ----------------------------------
from sklearn.metrics import classification_report, confusion_matrix
# show tuning results: display scores, best model's parameters, classification report and confusion matrix for TRAINING data
print(grid_model2.best_score_)

rf_pred = grid_model2.predict(lof_X_train)

print(classification_report(y_true = lof_y_train, y_pred = rf_pred))
print(confusion_matrix(y_true = lof_y_train, y_pred = rf_pred))


# #### <center>Record the best hyperparameters and performance resulting from this pipeline.</center>
# Detail Hyperparameters and Results below  (Question #E208)
# ---------------------------------------------

This model performed slightly better on the CV at 95%.  However, the Precision, Recall, and F1 were all lower ranging from 95% to 98%.  The first pipeline had tighter metrics for Precision, Recall, and F1.

Variance Threshold was needed in addition to SelectKBest, due to the F ANOVA of SelectKBest.  I was getting low variance warnings during CV search.  Variance Threhold helped to filter before using SelectKBest.

Below is the full metrics:
('var_thresh', VarianceThreshold(threshold=0.2)),
                ('scale', StandardScaler()), ('select', SelectKBest(k=11)),
                ('rfc',
                 RandomForestClassifier(max_features='log2',
                                        min_samples_split=3, n_estimators=300,
                                        random_state=42))

This pipeline notably had a significant increase in false positives compared to the first pipeline.
# ## <span style="background: yellow;">Commit your code!</span> 

# ---
# 
# ### Your 3rd pipeline
#   * Anomaly detection
#   * Dimensionality reduction
#   * Model training/validation

# In[4]:


from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[5]:


# Add anomaly detection code  (Question #E209)
# ----------------------------------
def OneClass(X,y):
    ocs = OneClassSVM(kernel = 'linear')
    y_pred = ocs.fit_predict(X)
    return X[y_pred == 1], y[y_pred == 1]


# In[6]:


svm_X_train, svm_y_train = OneClass(X_train, y_train)
svm_X_train.shape


# In[7]:


# Add code for pipeline with feature selection and classification and hyperparameter tuning (Question #E210)
# ----------------------------------
pipe3 = Pipeline([
    ('scale', StandardScaler()),
    ('factor', FactorAnalysis()),
    ('neighbors', KNeighborsClassifier())
])


# In[18]:


param_grid3 = {
    'factor__n_components':np.linspace(1,6,num=4,dtype=int),
    'neighbors__n_neighbors':np.linspace(5,10,num=5,dtype=int),
    'neighbors__weights':['uniform', 'distance']
}


# In[19]:


from sklearn.model_selection import RandomizedSearchCV

grid_model3 = RandomizedSearchCV(pipe3, param_distributions = param_grid3, n_iter=20, n_jobs = 5)


# In[20]:


grid_model3.fit(svm_X_train, svm_y_train)
print(grid_model3.best_estimator_)


# In[21]:


# Give an evaluation  (Question #E211)
# ----------------------------------
from sklearn.metrics import classification_report, confusion_matrix
# show tuning results: display scores, best model's parameters, classification report and confusion matrix for TRAINING data
print(grid_model3.best_score_)

k_pred = grid_model3.predict(svm_X_train)

print(classification_report(y_true = svm_y_train, y_pred = k_pred))
print(confusion_matrix(y_true = svm_y_train, y_pred = k_pred))


# #### <center>Record the best hyperparameters and performance resulting from this pipeline.</center>
# Detail Hyperparameters and Results below  (Question #E212)
# ---------------------------------------------

This pipeline performed slightly worse on the CV compared to the others at 90%.  However, The accuracy, precision, recall, and F1 on the training data were near 100%.  This may indicate overfitting to the training data, we will see how it does on the testing data.


Pipeline parameters below:
Pipeline(steps=[('scale', StandardScaler()),
                ('factor', FactorAnalysis(n_components=6)),
                ('neighbors',
                 KNeighborsClassifier(n_neighbors=8, weights='distance'))])

# ---
# 
# ## Compare these three pipelines and discuss your findings
# 
# Do NOT fit here again, just use the predict function with the test data and report which of the three pipelines perform the best. It is up to you what you deem "best". 

# In[12]:


# Testing metrics for pipeline 1
# Test vars are already outlier filtered
xgb_pred_test = grid_model.predict(iso_X_test)
print('Pipleine 1 Test Metrics:', '\n', classification_report(y_true = iso_y_test, y_pred = xgb_pred_test),
      '\n', confusion_matrix(y_true = iso_y_test, y_pred = xgb_pred_test))


# In[16]:


# Testing metrics for pipeline 2
#lof_X_test, lof_y_test = lof(X_test, y_test, 10)
rf_pred_test = grid_model2.predict(lof_X_test)
print('Pipeline 2 Test Metrics:', '\n', classification_report(y_true = lof_y_test, y_pred = rf_pred_test),
      '\n', confusion_matrix(y_true = lof_y_test, y_pred = rf_pred_test))


# In[22]:


# Testing metrics for pipeline 3
svm_X_test, svm_y_test = OneClass(X_test, y_test)
k_pred_test = grid_model3.predict(svm_X_test)
print('Pipeline 3 Test Metrics:','\n', classification_report(y_true = svm_y_test, y_pred = k_pred_test),
      confusion_matrix(y_true = svm_y_test, y_pred = k_pred_test))

# Write your analysis in this cell (Question #E213)
# ----------------------------------


# show comparison results: 

# display scores, classification reports and confusion matrices for three pipelines using TESTING data

Pipeline 1 performs the best with the highest precision, recall, and f1 scores.  I will proceed with Pipeline 1.



# ## <span style="background: yellow;">Commit your code!</span> 

# ---
# 
# ### Pickle the required pipeline/models and save them to use in Part 3.
# 
# Save your best pipeline and the anomaly detection model. Do NOT save the grid object, save the **best_estimator_** of the grid object. 

# In[15]:


model = grid_model.best_estimator_
joblib.dump([model, iso_forest], 'model_best_estimators.pkl')


# You should have made a few commits so far of this project.  
# **Definitely make a commit of the notebook now!**  
# Comment should be: `Final Project, Checkpoint - Pipelines done`
# 

# # Save your notebook!
# ## Then `File > Close and Halt`
