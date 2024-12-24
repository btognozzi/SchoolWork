#!/usr/bin/env python
# coding: utf-8

# # Spawn this notebook in the PyTorch-CPU container
# 
# The reason being that PyTorch is not installed in the CaseStudy-Capstone container.

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import sqlalchemy
import getpass
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Setting random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# In[2]:


# Create database connection string
database='casestdysu24t01'  
username='bat5h8'              
host='pgsql.dsa.lan'        
password=getpass.getpass()

connection = 'postgresql://'+username+':'+password+'@'+host+'/'+database


# In[3]:


engine = sqlalchemy.create_engine(connection)


# In[4]:


with engine.connect() as connection:

    query2= """
    SELECT
    district.dist_name, district.geoid, district.dist_code, acperf.year,
    --Table: acperf 
    acperf.act_english_score, acperf.act_math_score, acperf.act_reading_score, acperf.act_science_score, 
     acperf.w_grad_rt_4yr_cohort, acperf.disadv_grad_rt_4yr_cohort,
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
    mctv.mctv_percent, mctv.mctv_minmax,

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
    

del password    

df.shape


# # Data Carpentry
# 
# Will identify null columns, their values and drop necessary columns.

# In[5]:


nulls = []
for col in df.columns:
    if df[col].isnull().sum() > 0:
        null_count = df[col].isnull().sum()
        nulls.append('{}:{}'.format(col, null_count))
        
nulls


# In[6]:


# Dropping 2023 because it is the year we have the least data on
df = df[df['year'] != 2023]

nan_percentages = df.isna().mean() * 100

# Dropping columns with more than 20% data missing
columns_with_high_nan = nan_percentages[nan_percentages > 20].index

print("Columns with more than 20% NaN values:")
for col, percentage in nan_percentages.items():
    if percentage > 20:
        print(f"{col}: {percentage:.2f}%")


# In[7]:


df = df.drop(columns=columns_with_high_nan)


# In[8]:


df = df.dropna()
df.shape


# In[9]:


# Isolating the years and district codes if needed at a later time
id_df = df[['year', 'dist_code', 'geoid', 'dist_name']]


# In[10]:


# Dropping obvious columns which have no value to analysis
unecessary_cols = ['year', 'dist_code', 'dist_name', 'geoid', 'dist_code']
df.drop(columns = unecessary_cols, inplace = True)


# In[11]:


df.info()


# # Tensor Preparations
# 1. Create feature dataset and spliting each target variable since there are two
# 2. Normalize by dividing by the total number of rows in the dataset
# 3. Convert to Numpy array and then tensors
# 4. Combine and create the target1 and target2 datasets
# 5. Split into train and validation datasets

# In[12]:


data = df.drop(columns = ['mctv_minmax', 'mctv_percent'])
target1 = df[['mctv_minmax']]
target2 = df[['mctv_percent']]


# In[13]:


# Convert to tensors
data = data / np.max(data)
target1 = target1 / np.max(target1)
target2 = target2 / np.max(target2)

data = data.to_numpy()
target1 = target1.to_numpy()
target2 = target2.to_numpy()

data_tensor = torch.tensor(data, dtype = torch.float32)
target1_tensor = torch.tensor(target1, dtype = torch.float32).view(-1,1)
target2_tensor = torch.tensor(target2, dtype = torch.float32).view(-1,1)


# In[14]:


# Combine tensors into a TensorDataset and split into train and validation datasets
dataset = TensorDataset(data_tensor, data_tensor, target1_tensor, target2_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False)


# # Creating Autoencoder Neural Network Model
# 
# 1. MultiTask Network:
# 
#     The MultiTaskAutoencoder predicts two targets simultaneously. This multitask setup leverages shared representations to improve predictive performance and computational efficiency.
# 
# 
# 2. Encoder Layer:
# 
#     The encoder consists of three linear layers with 128, 64, and encoding_dim neurons, respectively. Each linear layer is followed by a LeakyReLU activation function. The LeakyReLU activation is chosen to prevent the "dying ReLU" problem, where neurons can become inactive for inputs near zero.
#     The input dimension (input_dim) corresponds to the number of features in the dataset. The sequential arrangement ensures that each neuron's output is fed into the next layer.
# 
# 
# 3. Decoder Layer:
# 
#     The decoder mirrors the encoder with three linear layers, converting the encoded representation back to the original input dimension. The final layer uses a sigmoid activation function to ensure the output values are between 0 and 1, which is typical for reconstruction tasks.
# 
# 
# 4. Multitask Prediction:
# 
#     Two separate linear layers (predictor1 and predictor2) are used to predict the two target values from the encoded representation. This setup allows the model to learn shared features while optimizing for both targets.
# 
# 
# 5. Model Functionality:
# 
#     During the forward pass, the input data is encoded into a lower-dimensional representation, decoded back to the original dimension, and simultaneously used for predicting the two targets. The autoencoder effectively reduces the dataset to a compact representation, facilitating efficient analysis and prediction.

# In[15]:


class MultiTaskAutoencoder(nn.Module):
    
    """
    This will define the model and inputs for the NN.  
    This is a mutli-task setup, so it is predicting both targets to see which is better.
    """
    
    def __init__(self, input_dim, encoding_dim):
        super(MultiTaskAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.Linear(128,64),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.Linear(64, encoding_dim),
            nn.LeakyReLU(negative_slope = 0.01)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        self.predictor1 = nn.Linear(encoding_dim, 1)
        self.predictor2 = nn.Linear(encoding_dim, 1)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        predicted1 = self.predictor1(encoded)
        predicted2 = self.predictor2(encoded)
        return decoded, predicted1, predicted2
    


# In[16]:


# initalize the model - define input dimension and the encoding dimension
# Played around with the best encoding dimension
input_dim = data.shape[1]
encoding_dim = 35
model = MultiTaskAutoencoder(input_dim, encoding_dim)


# # Setting up the Training Loop
# 
# 1. Define the reconstruction and prediction loss functions, these were set to MSE Loss functions.
# 
# 2. Define the optimizer, I explored many different optimzers (Adam, Rprop, RMSprop, SGD), I chose Adam Weight or AdamW.  This differs from Adam due to the Weight decay argument, which will apply a weight decay penalization to help with convergence (will incentivize simpler model paths).
# 
# 3. Define a learning rate scheduler, this will allow the learning rate to be dynamically adjusted as the network is trained.  I explored many different ones, but settled on the reduce on plateau.  This will decrease the learning rate if the model is getting consecutively stuck on a certain metric, in my case I had it monitor the validation loss.  With the change in learning rate it will help the model to try and find a new path and contiue until it settles again, so on and so forth.
# 
# 4. Within each epoch, the model is trained, the data batch is loaded, and the loss reconstruction and predicitons are calculated.  I also concurrently loaded the validation dataset at the same time to monitor overfitting real time as each epoch progress and the output is printed for each epoch.

# In[17]:


# Creating training loop for NN

def train_loop(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion_reconstruction = nn.MSELoss()
    criterion_prediction = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay = 0.01, betas = (0.82, 0.94))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data_batch, target_reconstruction, target1, target2 in train_loader:
            optimizer.zero_grad()
            reconstructed, predicted1, predicted2 = model(data_batch)
            loss_reconstruction = criterion_reconstruction(reconstructed, target_reconstruction)
            loss_prediction1 = criterion_prediction(predicted1, target1)
            loss_prediction2 = criterion_prediction(predicted2, target2)
            loss = 0.5 * loss_reconstruction + 0.25 * loss_prediction1 + 0.25 * loss_prediction2
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_batch, target_reconstruction, target1, target2 in val_loader:
                reconstructed, predicted1, predicted2 = model(data_batch)
                loss_reconstruction = criterion_reconstruction(reconstructed, target_reconstruction)
                loss_prediction1 = criterion_prediction(predicted1, target1)
                loss_prediction2 = criterion_prediction(predicted2, target2)
                loss = 0.5 * loss_reconstruction + 0.25 * loss_prediction1 + 0.25 * loss_prediction2
                val_loss += loss.item() * data_batch.size(0)

            val_loss /= len(val_loader.dataset)
            
            scheduler.step(val_loss)

            print(f'Epoch [{epoch+1} / {num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
    return model


# In[18]:


# Set up and pass necessary arguments into training function, using train loader and val loader created above
# Epochs and inital learning rate are specified
model = train_loop(model, train_loader, val_loader, 300, 0.001)


# # Output Analysis
# 
# Neural networks are a "black box" in design and it is hard to interpret the meaning of the network and how it is making its decision.  However, extracting the weights of the encoder can provide some insight into which variables have the greatest effect on the target variable.  In addition, passing the results into a clustering algorithm can also provide some insights into the functionality of the network.
# 
# 1. Access and extract the weights of the encoder function in the model.
# 2. Extract and input into a Numpy array and convert to pandas df using the column names from the data frame used to create the tensors.
# 3. Subset the weights by the abosolute value mean and sort the values in descending order.
# 4. Create a bar graph to output the most important features.
# 
# It is important to stress that we wont be able to tell if there is a positive or negative relationship between these variables and the target variable.  However, the weights do provide some base level of knowledge into what the network thinks are the most important variables with respect to the target variables.

# In[19]:


# Access encoder weights
encoder_layer = model.encoder[0]  # Access the first layer of the encoder
encoder_weights = encoder_layer.weight.detach().cpu().numpy()

column_names = df.drop(columns=['mctv_minmax', 'mctv_percent']).columns

# Create a DataFrame to inspect the encoder weights
encoder_weights_df = pd.DataFrame(encoder_weights, columns=column_names)
encoder_weights_df


# In[20]:


feature_importance = encoder_weights_df.abs().mean(axis=0)


# In[21]:


sorted_features_top20 = feature_importance.sort_values(ascending=False).head(20)


# In[22]:


plt.figure(figsize=(10, 6))
sorted_features_top20.plot(kind='bar')
plt.title('Top 20 Feature Importances based on Encoder Weights')
plt.xlabel('Feature')
plt.xticks(rotation = 45, horizontalalignment = 'right')
plt.ylabel('Importance')
plt.show()


# We can see that the top 20 most important features from the network range from white graduation rate to area (size) of the school district, to teach certifications, and other demographic features.  It seems like the target variable is most associated with demographic features.  These are important to identify as they will tell us alot about the clusters that we want to isolate.  We will want to understand all of the information related to the clusters as it can help to interpret and extract important insights within each cluster.  Among the most actionable is teach certifications, which can tell us a lot about a school district, while certifications are a teacher dependent decision, it can tell us a lot about a school district based on how many certifications teachers have.  The poverty features tell a descriptive story about the given school district and while maybe not directly actionable, policymakers could enact laws to help out these populations.

# # Cluster Analysis
# 
# This is still a work in progress...

# In[23]:


# Import using Mini Batch K Means algorithm for cluster analysis
from sklearn.cluster import MiniBatchKMeans


# In[24]:


# Extract encoded features from model
model.eval()
with torch.no_grad():
    encoded_features = model.encoder(data_tensor).numpy()


# In[28]:


# Cluster set up
# Using Mini Batch K Means, since normal K Means instantly kills the kernel
# Mini Batch will sacrifice some precision in the effort for more computationally efficiency

k = 5

kmeans = MiniBatchKMeans(n_clusters = k, batch_size = 3037,random_state = 42)
kmeans.fit(encoded_features)
labels = kmeans.labels_


# In[29]:


cluster_centers = kmeans.cluster_centers_

df['cluster'] = labels
cluster_summary = df.groupby('cluster').mean()
cluster_summary.T


# In[31]:


# For better interpretablility in the graphing of clusters, I will use PCA to reduce to 2 components and then graph
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca_result = pca.fit_transform(encoded_features)

plt.scatter(pca_result[:,0], pca_result[:,1], c = labels, cmap = 'viridis')
plt.title('PCA of Clusters in Latent Space')
plt.show()


# In[36]:


[clus.mean() for clus in cluster_centers]


# In[39]:


pca2 = PCA(n_components = 3)
pca2_result = pca.fit_transform(encoded_features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3D')
scatter = ax.scatter(pca2_result[:,0], pca2_result[:,1], pca2_result[:,2], c=labels, cmap='viridis', s=50)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('3D PCA Clustering')
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

plt.show()


# In[45]:


from mpl_toolkits.mplot3d import Axes3D

pca2 = PCA(n_components = 3)
pca2_result = pca2.fit_transform(encoded_features)

df_pca = pd.DataFrame(data=pca2_result, columns=['PCA1', 'PCA2', 'PCA3'])
df_pca['cluster'] = labels

fig = plt.figure(figsize=(10, 8))
ax = Axes3D(fig)
scatter = ax.scatter(df_pca['PCA1'], df_pca['PCA2'], df_pca['PCA3'], c=labels, cmap='viridis', s=50)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

plt.show()


# Looks like there are lots of overlap between all of the clusters.  We can see from the cluster centers that the means are all around 0.5.  However, within there are some variaitions within each cluster.  It may be worthwhile to examine further what differentiates the outside clusters from the overlapped clusters to find out how they are similar and how they are different from the core groupings.  When looking at the 3D representation, the clusters do not appear to separate any further, except for the outlying points in each cluster, which can be examined further.
