#!/usr/bin/env python
# coding: utf-8

# In[1]:

import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import make_scorer, silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# In[2]:


def basic_cleaning(df):
    
    #drop 2023 due to lack of census data
    df = df[df['year'] != 2023]
    
    #Calculate percentage of NaN values in each column
    nan_percentages = df.isna().mean() * 100

    #Filter columns where NaN percentage is greater than 20%
    columns_with_high_nan = nan_percentages[nan_percentages > 20].index
    
    #drop columns in list
    df = df.drop(columns=columns_with_high_nan)
    
    #Create numeric dataframe for future analytics
    num_df = df.select_dtypes(include=[int, float]).dropna()
    num_df = num_df.drop(columns=['year'], axis=1)
    
    #Create char dataframe for future identification 
    char_df = df.select_dtypes(include=[object])

    #Including the 'year' column
    if 'year' in df.columns:
        char_df = char_df.join(df['year'])

    #Dropping rows with NaN values from num_df
    indices = num_df.index
    char_df = char_df.loc[indices]
    
    #Reset index for easy join later
    char_df = char_df.reset_index(drop=True)
    num_df = num_df.reset_index(drop=True)
    
    X = num_df.drop(columns=['mctv_percent'])
    y = num_df['mctv_percent']
    
    target_descr = y.describe()
    
    #Extract mean and std from describe output
    mean_y = target_descr['mean']
    std_y = target_descr['std']

    #Define the categories based on standard deviations
    y_categories = pd.cut(y,
                          bins=[-float('inf'), mean_y - 2*std_y, mean_y - std_y, mean_y + std_y, mean_y + 2*std_y, float('inf')],
                          labels=[1, 2, 3, 4, 5],
                          right=True,
                          include_lowest=True)

    #Replace y with the categorical values
    y_cat_stdev = y_categories.astype(int)
    
    # Calculate quantile values based on your target variable statistics
    quantile_values = [
        target_descr['min'],  # Lower boundary, based on min
        target_descr['25%'],  # Q1
        target_descr['50%'],  # Median (Q2)
        target_descr['75%'],  # Q3
        target_descr['max']   # Upper boundary, based on max
    ]

    # Define categories based on quantiles
    y_categories_q = pd.cut(y,
                         bins=quantile_values,
                         labels=[1, 2, 3, 4],
                         right=True,
                         include_lowest=True)

    y_cat_quant = y_categories_q.astype(int)


    
    print("returns: X, y, y_cat_stdev, y_cat_quant, char_df, num_df")
    
    return X, y, y_cat_stdev, y_cat_quant, char_df, num_df


# In[10]:


def cont_y_RF_param_tune(X_train, X_test, y_train, y_test):
    
    print("model: RandomForestRegressor, random_state=42 \n"
           "parameters: n_estimators - 100, 200 \n"
                       "max_depth - 10, 15, 20 \n")
    
    #Define the pipeline: Standardization -> Factor Analysis -> Random Forest
    pipeline = Pipeline([
        ('random_forest', RandomForestRegressor(random_state=42))  #Random Forest
    ])

    #Define the hyperparameters for the entire pipeline
    param_grid = {
        'random_forest__n_estimators': [100, 200],  #Number of trees for Random Forest
        'random_forest__max_depth': [10, 15, 20]  #Maximum depth of trees for Random Forest
    }

    #GridSearchCV for the entire pipeline
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=10, scoring='neg_mean_squared_error', verbose=3)
    search.fit(X_train, y_train)

    #Print the best parameters found by GridSearchCV
    print(f"Best Params: {search.best_params_}")

    #Evaluate the best model
    best_model = search.best_estimator_
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")
    
    return best_model


# In[12]:


def cont_y_FA_RF_param_tune(X_train, X_test, y_train, y_test):
    
    print("model: StandardScaler -> factor_analysis -> random_forest (regressor) , random_state=42, cv=3 \n"
           "parameters: n_components - 6,7,8 \n"
                       "n_estimators - 100, 200 \n"
                       "max_depth - 10, 15, 20 \n")
    
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('factor_analysis', FactorAnalysis()),
    ('random_forest', RandomForestRegressor(random_state=42))
    ])
    

    #Define the hyperparameters for the entire pipeline
    param_grid = {
        'factor_analysis__n_components': [6,7,8],
        'random_forest__n_estimators': [100, 200],  #Number of trees for Random Forest
        'random_forest__max_depth': [10, 15, 20]  #Maximum depth of trees for Random Forest
    }

    #GridSearchCV for the entire pipeline
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=5, scoring='neg_mean_squared_error', verbose=3)
    search.fit(X_train, y_train)

    #Print the best parameters found by GridSearchCV
    print(f"Best Params: {search.best_params_}")

    #Evaluate the best model
    best_model = search.best_estimator_
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")
    
    # Transform the data using the best pipeline
    X_train_transformed = best_model.named_steps['scaler'].transform(X_train)
    X_train_transformed = best_model.named_steps['factor_analysis'].transform(X_train_transformed)
    
    X_test_transformed = best_model.named_steps['scaler'].transform(X_test)
    X_test_transformed = best_model.named_steps['factor_analysis'].transform(X_test_transformed)
    
    # Combine the transformed training and test data
    X_combined_transformed = np.vstack((X_train_transformed, X_test_transformed))
    
    return best_model, X_combined_transformed


# In[13]:


def cat_y_RF_param_tune(X_train, X_test, y_train, y_test):
    
    print("model: RandomForestClassifier, random_state=42 \n"
           "parameters: n_estimators - 100, 200 \n"
                       "max_depth - 10, 15, 20 \n")
    
    pipeline = Pipeline([
        ('random_forest', RandomForestClassifier(random_state=42))  #Random Forest
    ])

    #Define the hyperparameters for the entire pipeline
    param_grid = {
        'random_forest__n_estimators': [100, 200],  #Number of trees for Random Forest
        'random_forest__max_depth': [10, 15, 20]  #Maximum depth of trees for Random Forest
    }

    #GridSearchCV for the entire pipeline
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=10, scoring='neg_mean_squared_error', verbose=3)
    search.fit(X_train, y_train)

    #Print the best parameters found by GridSearchCV
    print(f"Best Params: {search.best_params_}")

    #Evaluate the best model
    best_model = search.best_estimator_
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")

    #Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    #Confusion matrix and classification report for training set
    print("\nTraining Set Evaluation:")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_pred_train))
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred_train))

    #Confusion matrix and classification report for test set
    print("\nTest Set Evaluation:")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    return best_model


# In[14]:


def cat_y_FA_RF_param_tune(X_train, X_test, y_train, y_test):
    
    print("model: StandardScaler -> factor_analysis -> random_forest (classifier) , random_state=42, cv=3 \n"
           "parameters: n_components - 6,7,8 \n"
                       "n_estimators - 100, 200 \n"
                       "max_depth - 10, 15, 20 \n")
    
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('factor_analysis', FactorAnalysis()),
    ('random_forest', RandomForestClassifier(random_state=42))
    ])
    

    #Define the hyperparameters for the entire pipeline
    param_grid = {
        'factor_analysis__n_components': [6,7,8],
        'random_forest__n_estimators': [100, 200],  #Number of trees for Random Forest
        'random_forest__max_depth': [10, 15, 20]  #Maximum depth of trees for Random Forest
    }

    #GridSearchCV for the entire pipeline
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=5, scoring='neg_mean_squared_error', verbose=3)
    search.fit(X_train, y_train)

    #Print the best parameters found by GridSearchCV
    print(f"Best Params: {search.best_params_}")

     #Evaluate the best model
    best_model = search.best_estimator_
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")

    #Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    #Confusion matrix and classification report for training set
    print("\nTraining Set Evaluation:")
    print("\nTraining Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred_train))
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_pred_train))

    #Confusion matrix and classification report for test set
    print("\nTest Set Evaluation:")
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Transform the data using the best pipeline
    X_train_transformed = best_model.named_steps['scaler'].transform(X_train)
    X_train_transformed = best_model.named_steps['factor_analysis'].transform(X_train_transformed)
    
    X_test_transformed = best_model.named_steps['scaler'].transform(X_test)
    X_test_transformed = best_model.named_steps['factor_analysis'].transform(X_test_transformed)
    
    # Combine the transformed training and test data
    X_combined_transformed = np.vstack((X_train_transformed, X_test_transformed))

    return best_model, X_combined_transformed


# In[8]:


def extract_factors_top10_loadings(best_model, variable_df):

    # Extract feature importances
    factor_analysis = best_model.named_steps['factor_analysis']

    # Get the factor loadings
    factor_loadings = factor_analysis.components_

    # Assuming 'X' contains your original features
    feature_names = variable_df.columns

    # Create a DataFrame to store factor loadings
    factor_loadings_df = pd.DataFrame(factor_loadings, columns=feature_names)

    num_factors = factor_loadings_df.shape[0]  # Number of factors
    top_n = 10  # Number of top loadings to display

    for factor_idx in range(num_factors):
        print(f"Loadings for Factor {factor_idx + 1}:")
        factor_variables = factor_loadings_df.iloc[factor_idx].sort_values(ascending=False).head(top_n)
        for variable, loading in factor_variables.items():
            print(f"{variable}: {loading:.6f}")
        print()
        
    return None


# In[15]:


def perc_feat_importance_selection(best_model, variable_df, threshold):
    
    #Extract feature importances
    feature_importances = best_model.named_steps['random_forest'].feature_importances_

    #Create a DataFrame to organize feature importances with their corresponding names
    feature_importances_df = pd.DataFrame({'Feature': variable_df.columns, 'Importance': feature_importances})

    #Sort features by importance (descending)
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    #Calculate cumulative importance
    feature_importances_df['Cumulative Importance'] = feature_importances_df['Importance'].cumsum()

    #Determine a threshold for cumulative importance (e.g., 90%)
    threshold = threshold
    selected_features = feature_importances_df[feature_importances_df['Cumulative Importance'] <= threshold]['Feature'].tolist()

    #Subset your data to include only the selected features
    X_selected = variable_df[selected_features]

    #Print the selected features
    print("Selected Features:")
    print(selected_features)
    print("\nNumber of Selected Features:")
    print(len(selected_features))
    print('returns: X_selected df')
    
    return X_selected

def tune_kmeans(X, cluster_range=range(2, 11)):
    def silhouette_scorer(estimator, X):
        cluster_labels = estimator.fit_predict(X)
        return silhouette_score(X, cluster_labels)
    
    param_grid = {'n_clusters': cluster_range}
    kmeans = KMeans(random_state=42)
    grid_search = GridSearchCV(kmeans, param_grid, scoring=make_scorer(silhouette_scorer(kmeans, X)), cv=5, n_jobs=10)
    grid_search.fit(X)
    
    best_kmeans_params = grid_search.best_params_
    best_kmeans_score = grid_search.best_score_
    
    print("Best K-Means Params:", best_kmeans_params)
    print("Best K-Means Silhouette Score:", best_kmeans_score)

    return None

def tune_dbscan(X, eps_range=np.arange(0.1, 1.1, 0.1), min_samples_range=range(2, 11)):
    def silhouette_scorer(estimator, X):
        labels = estimator.fit_predict(X)
        if len(set(labels)) <= 1:  # Silhouette score is not defined for a single cluster or noise only
            return -1
        return silhouette_score(X, labels)
    
    param_grid = {'eps': eps_range, 'min_samples': min_samples_range}
    dbscan = DBSCAN()
    grid_search = GridSearchCV(dbscan, param_grid, scoring=make_scorer(silhouette_scorer(dbscan, X)), cv=5, n_jobs=10)
    grid_search.fit(X)
    
    best_dbscan_params = grid_search.best_params_
    best_dbscan_score = grid_search.best_score_
    
    print("Best DBSCAN Params:", best_dbscan_params)
    print("Best DBSCAN Silhouette Score:", best_dbscan_score)

    return None


def tune_kmeans_v2(X, cluster_range = range(2,11)):
    
    best_score = 10000
    best_params = None
    best_model = None
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters = n_clusters, random_state = 42, init = "k-means++")
        cluster_labels = kmeans.fit_predict(X)
        
        if len(np.unique(cluster_labels)) < 2:
            continue
            
        score = davies_bouldin_score(X, cluster_labels)
        
        if score < best_score:
            best_score = score
            best_params = {'n_clusters':n_clusters}
            best_model = kmeans
            
    print("Best K-Means Params:", best_params)
    print("Bets K-Means Davies Bouldin Score:", best_score)
    
    return best_model, best_params, best_score




def tune_kmeans_v3(X, cluster_range = range(2,11), n_init_range = range(3,10), 
                   max_iter_range = np.arange(300, 500, 50), tol_range = np.logspace(-9, -4, num = 12)):
    
    best_score = 10000
    best_params = None
    best_model = None
    
    for n_clusters in cluster_range:
        for n_init in n_init_range:
            for n_iter in max_iter_range:
                for tol in tol_range:
                    kmeans = KMeans(n_clusters = n_clusters, random_state = 42, init = "k-means++",
                                    n_init = n_init, max_iter = n_iter, tol = tol)
                    cluster_labels = kmeans.fit_predict(X)
        
                    if len(np.unique(cluster_labels)) < 2:
                        continue
            
                    score = davies_bouldin_score(X, cluster_labels)
        
                    if score < best_score:
                        best_score = score
                        best_params = {'n_clusters':n_clusters,
                                   'n_init':n_init,
                                   'max_iter':n_iter,
                                   'tol':tol}
                        best_model = kmeans
            
    print("Best K-Means Params:", best_params)
    print("Bets K-Means Davies Bouldin Score:", best_score)
    
    return best_model, best_params, best_score


def tune_dbscan_v2(X, eps_range=np.arange(0.0001, 0.0009, 0.0001), min_samples_range=range(50, 100, 10)):
    
    best_score = 10000
    best_params = None
    best_model = None
    
    for eps in eps_range:
        for sample in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=sample)
            cluster_labels = dbscan.fit_predict(X)
            
            unique_labels = np.unique(cluster_labels)
            
            if len(unique_labels) < 2 or (len(unique_labels) == 1 and -1 in unique_labels):
                continue
            
            labels_filtered = cluster_labels[cluster_labels != -1]
            X_filtered = X[cluster_labels != -1]
            
            if len(np.unique(labels_filtered)) < 2:
                continue
            
            score = davies_bouldin_score(X_filtered, labels_filtered)
        
            if score < best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': sample}
                best_model = dbscan
    
    if best_params:
        print("Best DBSCAN Params:", best_params)
        print("Best DBSCAN Davies Bouldin Score:", best_score)
    else:
        print("No valid clustering found within the given parameter range.")
    
    return best_model, best_params, best_score


def cluster_adjustment(cluster_range, df, name_df, filename):
    for cluster in cluster_range:
        kmeans = KMeans(n_clusters = cluster, init = "k-means++", random_state = 42, n_init = 5, tol = 1e-6)
        cluster_labels = kmeans.fit_predict(df)
        
        df['cluster_labels_{}'.format(cluster)] = cluster_labels
        
    merged_df = pd.merge(df, name_df, left_index = True, right_index = True)
    
    merged_df.to_pickle("/dsa/groups/casestudy2024su/01/{}.pkl".format(filename))
        
    return merged_df


def reduce_dimensionality(X, method):
    if method == 'pca':
        pca = PCA(n_components = 2)
        X_reduced = pca.fit_transform(X)
    elif method == 'tsne':
        tsne = TSNE(n_components = 2, random_state = 42)
        X_reduced = tsne.fit_transform(X)
        
    return X_reduced


def plot_clusters(X, labels, method, save_path = None):
    
    if X.shape[1] > 2:
        X_reduced = reduce_dimensionality(X, method)
    else:
        X_reduced = X
        
    
    df_plot = pd.DataFrame(X_reduced, columns = ['component_1', 'component_2'])
    df_plot['labels'] = labels
    
    plt.figure(figsize = (10,6))
    sns.scatterplot(data = df_plot, x = 'component_1', y = 'component_2', hue = 'labels', palette = 'Paired',
                    legend = 'full', alpha = 0.6, s = 100)
    
    if -1 in labels:
        noise_points = df_plot[df_plot['labels'] == -1]
        plt.scatter(noise_points['Component 1'], noise_points['Component 2'], color='black', 
                    label='Noise', s=100, alpha=0.6)
        
    plt.title("KMeans Clusters")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()









