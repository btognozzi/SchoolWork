import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
def iso_forest(X,y,contamination='auto'):
    forest = IsolationForest(max_samples=100, contamination=contamination, random_state=42)
    forest.fit(X)
    y_pred = forest.predict(X)
    return X[y_pred == 1], y[y_pred == 1]
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
