#!/usr/bin/env python
# coding: utf-8

# # Part 3: Unbiased Evaluation using a New Test Set
# 
# In this part, we are given a new test set that serves as the "truly unseen data" (`/dsa/data/all_datasets/back_order/Kaggle_Test_Dataset_v2.csv`). We can now take advantage of the entire smart sample that we created in Part I. 
# 
# * Load your best pipeline model and anomaly detector from Part 2. 
# * Load your balanced (smart) sample deom Part 1. 
# * Retrain the model with the entire balanced sample. (do NOT repeat the grid search)
# * Save the model. 
# * Test it with the "unseen" data. 

# ## Import modules as needed

# In[1]:


import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from custom_function import iso_forest, lead_time_cat, sales_forecast_cat, min_bank_cat


# ---
# 
# ## Load the balanced sample and the best pipeline and the anomaly detector

# In[2]:


X_train,y_train = joblib.load('sample-data.pkl')
model_pipeline, function = joblib.load('model_best_estimators.pkl')


# In[3]:


model_pipeline


# ---
# 
# ##  Retrain pipeline using the full balanced sample 
# 
# Use the full balanced sample to train the pipeline.

# In[4]:


pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA()),
    ('xgb', XGBClassifier())
])


# In[5]:


params = {
    'scale':model_pipeline[('scale')],
    'pca':model_pipeline[('pca')],
    'xgb':model_pipeline[('xgb')]
}


# In[6]:


# Add code below this comment  (Question #E301)
# ----------------------------------
iso_X, iso_y = iso_forest(X_train, y_train)

pipe.set_params(**params)

model = pipe.fit(iso_X, iso_y)


# ## Pickle and save the trained model and the anomaly detector 

# In[23]:


# Add code below this comment  
# -----------------------------
# previously pickled the anomaly detector, will pickle again
joblib.dump([model, iso_forest], 'final_model_anomaly.pkl')


# 
# ---
# 
# ## Load the test data and evaluate your model
# 
#  * `/dsa/data/all_datasets/back_order/Kaggle_Test_Dataset_v2.csv`
# 
# Remember:  
# * We need to preprocess this test data (**follow** the steps similar to Part I)
# 
# 
# * If you have fitted any normalizer/standardizer in Part 2, then you have to transform this test data using the same fitted normalizer/standardizer.

# In[2]:


trained_model, anomaly = joblib.load('final_model_anomaly.pkl')


# In[3]:


# Preprocess the given test set  (Question #E302)
# ----------------------------------

test_data = pd.read_csv('/dsa/data/all_datasets/back_order/Kaggle_Test_Dataset_v2.csv')
test_data.drop(columns = ['sku'], inplace = True)
test_data = test_data.dropna(axis=0)
lead_time_cat(test_data)
sales_forecast_cat(test_data)
min_bank_cat(test_data)

print(test_data.shape)
print(test_data.info())


# In[4]:


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
            
test_data = encoder(test_data)
print(test_data.info())


# In[5]:


# upon review of the preprocessing, I did not save the dummy categories created, so I will drop the lead time categories
X_test = test_data.drop(columns = ['went_on_backorder', 'lead_time_cat_high', 'lead_time_cat_low', 'lead_time_cat_medium'])
y_test = test_data['went_on_backorder']


# In[6]:


# filtering the outliers for comparison
X_test_iso, y_test_iso = iso_forest(X_test, y_test)


# We can now predict and evaluate with the preprocessed test set. It would be interesting to see the performance with and without outliers removal from the test set. 
# 
# Report confusion matrix, precision, recall, f1-score, accuracy, and other measures (if any). 

# In[7]:


# Add code below this comment  (Question #E303)
# ----------------------------------
from sklearn.metrics import classification_report, confusion_matrix

final_pred = trained_model.predict(X_test)
final_pred_out = trained_model.predict(X_test_iso)

print('Classification Report:', '\n', classification_report(y_true = y_test, y_pred = final_pred))
print("")
print('Confusion Matrix:', '\n', confusion_matrix(y_true = y_test, y_pred = final_pred))
print("")
print('Classification Report Outliers Removed:', '\n', classification_report(y_true = y_test_iso, y_pred = final_pred_out))
print('')
print('Confusion Matrix:', '\n', confusion_matrix(y_true = y_test_iso, y_pred = final_pred_out))


# In[14]:


# I will report out the precision, recall, and f1.  I will change the average to 'weighted' to account for the imbalanced dataset
from sklearn.metrics import precision_score, recall_score, f1_score

def metrics(y_pred, y_true):
    pre = print('Precision:',precision_score(y_true = y_true, y_pred = y_pred, average = 'weighted'))
    rec = print('Recall:',recall_score(y_true = y_true, y_pred = y_pred, average = 'weighted'))
    f1 = print('F1:',f1_score(y_true = y_true, y_pred = y_pred, average = 'weighted'))
    return pre, rec, f1


# In[15]:


print("Non Outlier Removal:")
print(metrics(final_pred, y_test))
print('')
print('Outlier Removal:')
print(metrics(final_pred_out, y_test_iso))


# ---
# 
# ## Conclusion
# 
# Comment on the performance of your model: take a look at the project notes to see what you should report here. 
# Write a summary of your processing and an analysis of the model performance  
# (Question #E304)
# ----------------------------------

Initially, to enhance data quality, I opted to remove all NA values. Subsequently, I conducted feature engineering on several variables, including sales, forecast, min_bank, and lead_time, creating categorical variables. This step proved instrumental in uncovering latent relationships within the dataset.

During cross-validation, the model showcased superior performance on unseen data compared to the training set. This was achieved through training on a balanced dataset and adjusting the metrics to reflect the weighted nature of the unseen data, resulting in significantly improved performance.

In considering sampling techniques, a hybrid approach involving both under-sampling and over-sampling proved effective. However, I believe Condensed Nearest Neighbors could offer a more refined solution compared to random under-sampling. Additionally, techniques such as SMOTE or ADASYN, if memory constraints permit, could further enhance model effectiveness, with ADASYN offering synthetic point creation while preserving variance crucial for the minority class.

Exploring feature engineering possibilities for other variables may yield further improvements. Although outlier removal using Isolation Forest had minimal impact on metrics, considering alternatives like LOF, which performed well during cross-validation, could be beneficial.

The choice of XGBoostClassifier proved suitable for handling imbalanced datasets, contributing to the model's success in this problem domain. Looking ahead, opportunities for refinement lie in continued exploration of feature engineering, sampling techniques, and more effective outlier removal methods.
# ---
# 
# ## Reflection
# 
# Imagine you are data scientist that has been tasked with developing a system to save your 
# company money by predicting and preventing back orders of parts in the supply chain.
# 
# Write a **brief summary** for "management" that details your findings, 
# your level of certainty and trust in the models, 
# and recommendations for operationalizing these models for the business. take a look at the project notes to see what you should report here. 
# Write your answer here:  
# (Question #E305)
# ---------------------------------- 

The trained model has demonstrated strong performance and is suitable for deployment in a production environment. Precision, recall, and the F1 score indicate positive predictive power and sensitivity, striking a balance between the two. While the confusion matrix reveals minimal false negatives, it also highlights a notable presence of false positives.

Considering these factors collectively, I am confident in the model's efficacy for operational use. It prioritizes minimizing false negatives, crucial for averting costly back orders and maintaining customer satisfaction. However, this focus on reducing false negatives entails an increase in false positives, necessitating additional time and resources for verification.

If the decision is made to implement the model in production, it may require hiring additional inventory staff to manage the verification process. The associated labor costs could be offset by the streamlined supply chain workflow facilitated by the model, eliminating the need for rush orders and ensuring timely delivery of critical parts.

This trade-off in model performance underscores the need for short-term process improvements to enhance workflow efficiency. However, in the long run, I am confident that the model will justify the added labor costs by effectively identifying critical parts at risk of back orders.

Overall, I believe the model is well-suited for production use, even with the necessary resources for managing false positives. Enhancing customer satisfaction is paramount for building a strong brand and fostering trust with customers, positioning us as a reliable parts supplier in the market.

In conclusion, I anticipate that this model will yield meaningful and reliable predicitions, contributing to the efficiency of our supply chain.  While additional staffing may be required to address false positives, the resulting streamlined operations offer potential advantages over our competitors, further enhancing customer relationships and market position.
# # Save your notebook!
# 
# ## Commit and push. 
# 
# 
