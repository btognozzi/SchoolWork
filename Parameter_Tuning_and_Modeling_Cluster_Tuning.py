#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import Pipeline_Tuning_Functions_Custom_BT as cf
import inspect
import getpass
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# In[ ]:


mypass = getpass.getpass()
usern = 'bat5h8'


# In[3]:


# Connect to the database
try:
    # Establish the connection
    connection = psycopg2.connect(
        database='casestdysu24t01',  
        user=usern,               # Replace with your actual username
        host='pgsql.dsa.lan',        
        password=mypass              # Use the password entered by the user
    )
    print('Good connection')

    # Optionally, you can execute a simple query to test the connection
    cursor = connection.cursor()
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected to - ", record, "\n")

    # Query for bulk data grab
    query2= """
        SELECT
        district.dist_name, district.geoid, district.dist_code, acperf.year,
        --Table: acperf 
        acperf.act_english_score, acperf.act_math_score, acperf.act_reading_score, acperf.act_science_score, 
        acperf.disadv_grad_rt_4yr_cohort,
        acperf.grad_fu_college_pct, acperf.grad_fu_employed_pct,

        --Table: attendance
        attendance.proportional_attendance_frl_pct,
        attendance.proportional_attendance_iep_pct, attendance.fallenrollment, attendance.additionalenrollment,
        attendance.transfers, attendance.mobilityrate,

        --Table: census
        census.pctunder18, census.pcthouseholder, census.pctspouse,
        census.pctoppsexspouse, census.pctsamesexspouse, census.pctunmarriedpartner, census.pctoppsexpartner,
        census.pctsamesexpartner, census.pctownchild_under18, census.pctfamilykidslt18,
        census.pctmarrdcplkidslt18,census.pctsingledads,census.pctsinglemoms,

        --Table: district
        district.area_sqmil,

        --Table: faculty
        faculty.administrator_salary_average, faculty.teacher_salary_avg, faculty.teacher_salary_avg_reg_term,
        faculty.teacher_average_years_exp, faculty.teacher_mast_degree_percent,

        --Table: finance
        finance.average_daily_attendance, finance.current_expenditure,
        finance.current_expenditure_per_average_daily_attendance, finance.percent_federal,
        finance.percent_local, finance.percent_state, finance.total_expenditure,

        --Table: income
        income.fam, income.fam_mean_income,
        income.fam_med_income, income.famless_49999, income.fam50000_99999, income.fam100000_more, income.hh,
        income.hh_mean_income, income.hh_med_income, income.hhless_34999, income.hh35000_99999, income.hh100000_more,
        income.nfhh, income.nfhh_mean_income, income.nfhhless_14999, income.nfhh15000_74999, income.nfhh75000_more,
        income.mcf, income.mcfless_74999, income.mcf75000_149999, income.mcf150000_more,

        --Table: mctv
        mctv.mctv_percent,

        --Table: mshaa
        mshaa.students_participated,

        --Table: poverty
        poverty.pop,
        poverty.pct_bpl_pop, poverty.pct_bpl_pop_m, poverty.pct_bpl_pop_f, poverty.pct_bpl_pop_under18,
        poverty.pct_bpl_pop_white, poverty.pct_bpl_pop_aa, poverty.pct_bpl_pop_asian, poverty.pct_bpl_pop_other_race,
        poverty.pct_bpl_pop_edu_less_hs_grad, poverty.pct_bpl_pop_edu_hs_grad, poverty.pct_bpl_pop_edu_some_college,
        poverty.pct_bpl_pop_edu_college_grad,

        --Table: student_ratio
        student_ratio.students_per_admin_ratio,student_ratio.students_per_classrm_tch_ratio,
        student_ratio.students_per_teacher_ratio,

        --Table: teacher_cert
        teacher_cert.educator_reg_cert_pct,teacher_cert.educator_spec_cert_pct, teacher_cert.educator_no_cert_pct,
        teacher_cert.highly_qualified_crs_qualified,teacher_cert.highly_qualified_courses_pct

        FROM district
        FULL JOIN acperf ON district.dist_code = acperf.county_district_code
        FULL JOIN mctv ON acperf.year = mctv.year AND acperf.county_district_code = mctv.district_code
        FULL JOIN faculty ON acperf.year = faculty.year AND acperf.county_district_code = faculty.district_code
        FULL JOIN student_ratio ON acperf.year = student_ratio.year AND acperf.county_district_code = student_ratio.district_code
        FULL JOIN teacher_cert ON acperf.year = teacher_cert.year AND acperf.county_district_code = teacher_cert.district_code
        FULL JOIN finance ON acperf.year = finance.year AND acperf.county_district_code = finance.district_code
        FULL JOIN attendance ON acperf.year = attendance.year AND acperf.county_district_code = attendance.district_code
        FULL JOIN discipline ON acperf.year = discipline.year AND acperf.county_district_code = discipline.district_code
        FULL JOIN income ON district.geoid = income.geoid AND acperf.year = income.year
        FULL JOIN poverty ON district.geoid = poverty.geoid AND acperf.year = poverty.year
        FULL JOIN census ON district.geoid = census.geoid
        FULL JOIN mshaa ON district.dist_code = mshaa.district_code AND acperf.year = mshaa.year
        ORDER BY district.dist_code, acperf.year;
    """

    df = pd.read_sql(query2, connection)
    print('df obtained')

    # Close the cursor and connection
    cursor.close()
    connection.close()

except Exception as e:
    print(f"Error during connection: {e}")


# In[4]:


# Get all functions in the module
functions_list = inspect.getmembers(cf, inspect.isfunction)

# Print function names and their parameters
for function_name, function_obj in functions_list:
    print(f"Function: {function_name}")
    signature = inspect.signature(function_obj)
    for param in signature.parameters.values():
        print(f"    Parameter: {param.name}")
    print()


# In[5]:


X, y, y_cat_stdev, y_cat_quant, char_df, num_df = cf.basic_cleaning(df)


# In[6]:


X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(X, y, test_size=0.2, random_state=42)

print('X_train shape:', X_train_cont.shape)
print('X_test shape:', X_test_cont.shape)
print('y_train shape:', y_train_cont.shape)
print('y_test shape:', y_test_cont.shape)


# In[7]:


X_train_cat_sd, X_test_cat_sd, y_train_cat_sd, y_test_cat_sd = train_test_split(X, y_cat_stdev, test_size=0.2, stratify=y_cat_stdev, random_state=42)

print('X_train shape:', X_train_cat_sd.shape)
print('X_test shape:', X_test_cat_sd.shape)
print('Unique values and counts in y_train:')
print(y_train_cat_sd.value_counts())
print('Unique values and counts in y_test:')
print(y_test_cat_sd.value_counts())
        


# In[8]:


X_train_cat_q, X_test_cat_q, y_train_cat_q, y_test_cat_q = train_test_split(X, y_cat_quant, test_size=0.2, stratify=y_cat_quant, random_state=42)

print('X_train shape:', X_train_cat_q.shape)
print('X_test shape:', X_test_cat_q.shape)
print('Unique values and counts in y_train:')
print(y_train_cat_q.value_counts())
print('Unique values and counts in y_test:')
print(y_test_cat_q.value_counts())


# ## Continuous MCTV, RF only

# In[9]:


model_cont_RF = cf.cont_y_RF_param_tune(X_train_cont, X_test_cont, y_train_cont, y_test_cont)


# In[10]:


X_select_cont_rf = cf.perc_feat_importance_selection(model_cont_RF, X, threshold=0.9)


# ## Continuous MCTV, FA and RF

# In[11]:


model_cont_FA_RF, X_trans_cont = cf.cont_y_FA_RF_param_tune(X_train_cont, X_test_cont, y_train_cont, y_test_cont)


# ## Categorical-stdev, RF only

# In[12]:


model_cat_sd_RF = cf.cat_y_RF_param_tune(X_train_cat_sd, X_test_cat_sd, y_train_cat_sd, y_test_cat_sd)


# In[13]:


X_select_cat_sd_rf = cf.perc_feat_importance_selection(model_cat_sd_RF, X, threshold=0.9)


# ## Categorical-qant, RF only

# In[14]:


model_cat_q_RF = cf.cat_y_RF_param_tune(X_train_cat_q, X_test_cat_q, y_train_cat_q, y_test_cat_q)


# In[15]:


X_select_cat_q_rf = cf.perc_feat_importance_selection(model_cat_q_RF, X, threshold=0.9)


# ## Categorical-stdev, FA and RF

# In[16]:


model_cat_sd_FA_RF, X_trans_sd = cf.cat_y_FA_RF_param_tune(X_train_cat_sd, X_test_cat_sd, y_train_cat_sd, y_test_cat_sd)


# ## Categorical-quant, FA and RF

# In[17]:


model_cat_q_FA_RF, X_trans_quant = cf.cat_y_FA_RF_param_tune(X_train_cat_q, X_test_cat_q, y_train_cat_q, y_test_cat_q)


# In[18]:


# For Zero Division Error for Davies Bouldin Score
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


# # K-Means Categorical-stdev, FA and RF

# In[19]:


kmeans_stdev_model, kmeans_stdev_params, kmeans_stdev_score = cf.tune_kmeans_v2(X_trans_sd)


# # Finer tuned K-Means Cat Stdev

# In[ ]:


# Did not change anything from above, so will skip now
kmeans_stdev_model2, kmeans_stdev_params2, kmeans_stdev_score2 = cf.tune_kmeans_v3(X_trans_sd)


# # DBSCAN Categorical-stdev, FA and RF

# In[ ]:


dbscan_stdev_fa_rf = cf.tune_dbscan_v2(X_trans_sd)


# # K-Means Categorical-quant, FA and RF

# In[ ]:


kmeans_cat_q_model, kmeans_cat_q_params, kmeans_cat_q_score = cf.tune_kmeans_v2(X_trans_quant)


# # DBSCAN Categorical-quant, FA and RF

# In[ ]:


dbscan_cat_q_fa_rf = cf.tune_dbscan_v2(X_trans_quant)


# # Visualize Clusters

# In[ ]:


# Create DFs for stdev and quant
stdev_column_names = ['factor_1', 'factor_2', 'factor_3', 'factor_4',
                      'factor_5', 'factor_6', 'factor_7', 'factor_8']

quant_column_names = ['factor_1', 'factor_2', 'factor_3', 'factor_4',
                      'factor_5', 'factor_6', 'factor_7']

df_stdev = pd.DataFrame(X_trans_sd, columns = stdev_column_names)
df_quant = pd.DataFrame(X_trans_quant, columns = quant_column_names)


# In[ ]:


cluster_range = [4,5,6]
filename = "stdev_clusters_4-6"
final_stdev = cf.cluster_adjustment(cluster_range, df_stdev, char_df, filename)
final_stdev.head()


# In[ ]:


filename = "quant_clusters_4-6"
final_quant = cf.cluster_adjustment(cluster_range, df_quant, char_df, filename)
final_quant.head()


# Now that the cluster labels are added to the dataset, we need to reduce the dimensonality for plotting purposes.

# In[ ]:


cf.plot_clusters(final_stdev.drop(columns = ['cluster_labels_4', 'cluster_labels_5', 'cluster_labels_6',
                                       'dist_name', 'dist_code', 'year', 'geoid']), final_stdev['cluster_labels_4'], 'tsne')


# In[ ]:


cf.plot_clusters(final_quant.drop(columns = ['cluster_labels_4', 'cluster_labels_5', 'cluster_labels_6',
                                       'dist_name', 'dist_code', 'year', 'geoid']), final_quant['cluster_labels_4'], 'tsne')


# In[ ]:




