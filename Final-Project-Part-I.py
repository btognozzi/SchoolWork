#!/usr/bin/env python
# coding: utf-8

# # Final Project Part I (Due: Saturday December 9, 2023 11:59pm
# 
# In this final project, you will work through the Database Design, ETL and Analytics phases using a multii-year Chicago crime data source. This final project is divided into two parts.  Part I fill focus on the database design and ETL.  Part II will focus on the analytics.
# 
# In this Part I you will conduct the following tasks:
# 
# 1. Reverse engineer an existing sourse RDBMS using metadata SQL queries to identify the table and attribute details necessary for creating tables and an entity-relationship diagram depecting the database logical structure. The source data is an SQLite database (you might consider producing and ERD for this Sqlite database for your internal use, but **NOT REQUIRED to turn-in**).
# 2. Implement a set of tables using DDL in your SSO dsa_student database schema on the postgres server that replicates the source database structure. **Be sure to critically examine the source database structure, columns, constraints, and relationships (Foreign Key References) for accuracy**.  Ensure you have required data types and use the same exact table names as specified for the destination pgsql database.
# 3. Create an Entity Relationship Diagram for the **destination postgresql "database" tables** (not the Sqlite db).
# 4. Establish connections to the source and destination databases.
# 5. Extract the source data from tables, Transform values as required and Load into the destination tables.
# 6. Validate the ETL process by confirming row counts in both source and destination database tables.
# 
# 
# Specific resourses and steps are listed below:

# ## Source SQLite Database
# 
# * Dataset URL: **/dsa/data/DSA-7030/cc23_7030.sqlite.db**
# * Data Dictionary: [pdf](./ChicagoData-Description.pdf)
# * [Chicago Crimes 2001-Present Dashboard](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present-Dashboard/5cd6-ry5g)
# 
# This SQLite database consists of a set of normalized relations populated with publically available Chicago crime data for the years 2001 to 2022.  

# ## Database exploration
# 
# The cells below provide SQL DML statments for examining the underlying metadata in the SQLite database that describes the table, column, and relationship details.  An initial connection and subsequent SQL statements are provided for acquiring the information necessary for reconstructing the table and relational structure in your postgres SSO database.

# In[1]:


#Load extention and connect to database
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:////dsa/data/DSA-7030/cc23_7030.sqlite.db')


# ## Explore the SQLite Tables List
# 
# This quiery simply lists the names of the database tables.  Here's a quick reference discussing the sqlite_master table showing its utility.  [sqlite_master table meta data](https://bw.org/2019/02/25/finding-meta-data-in-sqlite/) - explore what metadata is provided.

# In[2]:


get_ipython().run_cell_magic('sql', '', "SELECT distinct m.type, m.tbl_name --m.sql\nFROM sqlite_master AS m,\n     pragma_table_info(m.name) AS t\nWHERE m.type = 'table'\norder by m.name, t.pk DESC")


# ## Explore Column Details
# 
# The query below provdes the complete list of tables and their columns with important details.
# 
# * **tbl_name** = Name of the table
# * **name** = column name
# * **type** = declared data type
# * **notnull** = indicates column declared as NOT NULL
# * **pk** = indicates column is the primary key

# In[3]:


get_ipython().run_cell_magic('sql', '', "SELECT m.tbl_name, t.* --m.sql\n FROM pragma_table_info(m.tbl_name) t, sqlite_master m WHERE m.type='table';")


# ## Below query provdes the list of columns that are declared "unique" for referential integrity enforcement.
# 
# <u>Query Output Descriptions</u>
# * **name** = the table name begining at the "cc_" -- cc_case_location is table name.
# * **unique** = indicates the column is declared "unique"
# * **origin** = indicates the columns is declared as primary key
# * **name_1** = column name

# In[4]:


get_ipython().run_cell_magic('sql', '', 'select il.*,ii.* --,m.sql\n    from sqlite_master m, \n    pragma_index_list( m.name ) as il,\n    pragma_index_info(il.name) as ii')


# ## Explore Relationship Details (get foreign key references)
# 
# The below query exracts the details describing the foreign key referenes bewtween tables.
# 
# * **from_table** = the name of the one-side table
# * **from_column** = the name of the foreign key column in the one-side table
# * **to_table** = the name of the many-side reference table
# * **to_column** = the name of the foreign key column in the one-side reference table
# 
# These metadata can be translated to the necessary SQL statement to establish a relationship between tables:
# 
# ```SQL
# FOREIGN KEY (<from_column>) REFERENCES <to_table>(<to_column>)
# ```

# In[5]:


get_ipython().run_cell_magic('sql', '', 'SELECT \n    m.name as from_table, f.\'from\' as from_column, f.\'table\' as to_table, f.\'to\' as to_column --, m.sql\nFROM\n    sqlite_master m\n    JOIN pragma_foreign_key_list(m.name) f ON m.name != f."table"\nWHERE m.type = \'table\'\nORDER BY m.name\n;')


# ## Using the metadata from above:
# 
# ## Implement the required CREATE TABLE statements for establishing the Chicago Crime Database in your SSO dsa_student database.  
# 
# The SQL statement takes this form:
# 
# ```SQL
# CREATE TABLE SSO.cc23_tbl_name (
#  column_name_1 data_type <unqiue, not null>,
#  column_name_N data_type <unqiue, not null>,
#  PRIMARY KEY (<column_name>),
#  <FOREIGN KEY (from_column_name) REFERENCES <SSO.cc23_to_table_name>(to_column_name)
#  );
# ```
# 
# **The database tables, column names, and data types created in your SSO postgres server dsa_student database should be named exactly as they appear (with necessary modifications for any constraint or reference anomalies) in the ```cc23_7030.sqlite.db``` SQLite database.**
# 
# Use as many cells as desired.
#sql create table statements

CREATE TABLE bat5h8.cc23_iucr_codes (
 iucr_code varchar(10),
 iucr_index_code char,
 PRIMARY KEY (iucr_code)
)
 
CREATE TABLE bat5h8.cc23_iucr_codes_primary_descriptions (
 iucr_code varchar(10),
 iucr_primary_desc varchar(100),
 PRIMARY KEY (iucr_code),
 FOREIGN KEY (iucr_code) REFERENCES bat5h8.cc23_iucr_codes (iucr_code)
)

CREATE TABLE bat5h8.cc23_iucr_codes_secondary_descriptions (
 iucr_code varchar(10),
 iucr_secondary_desc varchar(100),
 PRIMARY KEY (iucr_code),
 FOREIGN KEY (iucr_code) REFERENCES bat5h8.cc23_iucr_codes (iucr_code)
)
 
CREATE TABLE bat5h8.cc23_fbi_nibrs_categories (
 fbi_nibrs_category_name varchar(50),
 PRIMARY KEY (fbi_nibrs_category_name)
)

CREATE TABLE bat5h8.cc23_fbi_nibrs_offense_categories (
 nibrs_offense_code varchar(10) NOT NULL,
 fbi_nibrs_category_name varchar(50),
 PRIMARY KEY (nibrs_offense_code),
 FOREIGN KEY (fbi_nibrs_category_name) REFERENCES bat5h8.cc23_fbi_nibrs_categories (fbi_nibrs_category_name)
)

CREATE TABLE bat5h8.cc23_nibrs_crimes_against (
 nibrs_crimes_against varchar(20) NOT NULL,
 PRIMARY KEY (nibrs_crimes_against)
)

CREATE TABLE bat5h8.cc23_cases (
 case_number varchar(20),
 incident_date timestamp,
 iucr_code varchar(10),
 nibrs_fbi_offense_code varchar(10),
 arrest boolean,
 domestic boolean,
 updated_on timestamp,
 PRIMARY KEY (case_number),
 FOREIGN KEY (iucr_code) REFERENCES bat5h8.cc23_iucr_codes (iucr_code),
 FOREIGN KEY (nibrs_fbi_offense_code) REFERENCES bat5h8.cc23_fbi_nibrs_offense_categories (nibrs_offense_code)
)

CREATE TABLE bat5h8.cc23_nibrs_fbicode_offenses (
 nibrs_offense_code varchar(10) NOT NULL,
 nibrs_offense_name varchar(100) NOT NULL,
 PRIMARY KEY (nibrs_offense_code)
)

CREATE TABLE bat5h8.cc23_nibrs_offenses_crimes_aginst (
 nibrs_crime_against varchar(20),
 nibrs_offense_code varchar(10) NOT NULL,
 PRIMARY KEY (nibrs_crime_against, nibrs_offense_code),
 FOREIGN KEY (nibrs_crime_against) REFERENCES bat5h8.cc23_nibrs_crimes_against (nibrs_crime_against),
 FOREIGN KEY (nibrs_offense_code) REFERENCES bat5h8.cc23_nibrs_fbicode_offenses (nibrs_offense_code)
)

CREATE TABLE bat5h8.cc23_case_location (
 case_number varchar(20),
 block varchar(100),
 location_description varchar(100),
 community_area integer,
 ward integer,
 distinct integer,
 beat integer,
 latitude real,
 longitude real,
 PRIMARY KEY (case_number),
 FOREIGN KEY (case_number) REFERENCES bat5h8.cc23_cases (case_number)
)

# # Connect to your SSO database using %sql magic or sqlAlchmey connection and implement your database structure (create table...)

# In[1]:


#implement tables in SSO database
import psycopg2
import sqlalchemy
import getpass

user = "bat5h8"
passwrd = getpass.getpass()
engine = sqlalchemy.create_engine('postgresql://{}:{}@pgsql.dsa.lan/dsa_student'.format(user, passwrd))
connection = engine.connect()

del passwrd


# In[2]:


# Dropped tables to correct column names
connection.execute("DROP TABLE bat5h8.cc23_nibrs_crimes_against CASCADE;")
connection.execute("DROP TABLE bat5h8.cc23_nibrs_offenses_crimes_aginst CASCADE;")


# In[4]:


# corrected column names
connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_iucr_codes (
 iucr_code varchar(10),
 iucr_index_code char,
 PRIMARY KEY (iucr_code)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_iucr_codes_primary_descriptions (
 iucr_code varchar(10),
 iucr_primary_desc varchar(100),
 PRIMARY KEY (iucr_code),
 FOREIGN KEY (iucr_code) REFERENCES bat5h8.cc23_iucr_codes (iucr_code)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_iucr_codes_secondary_descriptions (
 iucr_code varchar(10),
 iucr_secondary_desc varchar(100),
 PRIMARY KEY (iucr_code),
 FOREIGN KEY (iucr_code) REFERENCES bat5h8.cc23_iucr_codes (iucr_code)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_fbi_nibrs_categories (
 fbi_nibrs_category_name varchar(50),
 PRIMARY KEY (fbi_nibrs_category_name)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_fbi_nibrs_offense_categories (
 nibrs_offense_code varchar(10) NOT NULL,
 fbi_nibrs_category_name varchar(50),
 PRIMARY KEY (nibrs_offense_code),
 FOREIGN KEY (fbi_nibrs_category_name) REFERENCES bat5h8.cc23_fbi_nibrs_categories (fbi_nibrs_category_name)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_nibrs_crimes_against (
 nibrs_crime_against varchar(20) NOT NULL,
 PRIMARY KEY (nibrs_crime_against)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_cases (
 case_number varchar(20),
 incident_date timestamp,
 iucr_code varchar(10),
 nibrs_fbi_offense_code varchar(10),
 arrest boolean,
 domestic boolean,
 updated_on timestamp,
 PRIMARY KEY (case_number),
 FOREIGN KEY (iucr_code) REFERENCES bat5h8.cc23_iucr_codes (iucr_code),
 FOREIGN KEY (nibrs_fbi_offense_code) REFERENCES bat5h8.cc23_fbi_nibrs_offense_categories (nibrs_offense_code)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_nibrs_fbicode_offenses (
 nibrs_offense_code varchar(10) NOT NULL,
 nibrs_offense_name varchar(100) NOT NULL,
 PRIMARY KEY (nibrs_offense_code)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_nibrs_offenses_crimes_aginst (
 nibrs_crime_against varchar(20),
 nibrs_offense_code varchar(10) NOT NULL,
 PRIMARY KEY (nibrs_crime_against, nibrs_offense_code),
 FOREIGN KEY (nibrs_crime_against) REFERENCES bat5h8.cc23_nibrs_crimes_against (nibrs_crime_against),
 FOREIGN KEY (nibrs_offense_code) REFERENCES bat5h8.cc23_nibrs_fbicode_offenses (nibrs_offense_code)
)
"""
)

connection.execute(
"""
CREATE TABLE IF NOT EXISTS bat5h8.cc23_case_location (
 case_number varchar(20),
 block varchar(100),
 location_description varchar(100),
 community_area integer,
 ward integer,
 district integer,
 beat integer,
 latitude real,
 longitude real,
 PRIMARY KEY (case_number),
 FOREIGN KEY (case_number) REFERENCES bat5h8.cc23_cases (case_number)
)
"""
)


# In[5]:


connection.close()


# ## Construct and embed your Entity Relationship Diagram for your destination cc23_ postgress database
# 
# Upload your ERD image to the "final_project" folder and update the markdown below to display it here:
# 
# ![image.png](attachment:image.png)
# 

# # Perform the ETL of the source data to your SSO dsa_student Chicago Crime Database
# 
# * Establish a connection to the the SQLite source database using sqlAlchemy (best choice) - use identifiable name.
# * Peform ETL of the source data tables to the destination data tables incrementally (best choice) - use identifiable name.
#   * You may want to use pandas as the medium to ETL between the two databases -- **be patient!**
#      * it can easliy read "big" source sql table data
#      * hold data in a resizable data frame relative to computing resource constraints
#      * make any necessary transformations to data values
#      * write/load data to destination postgresql tables
#     

# In[6]:


import psycopg2
import sqlalchemy

engine = sqlalchemy.create_engine('sqlite:////dsa/data/DSA-7030/cc23_7030.sqlite.db')
connection = engine.connect()


# In[7]:


import pandas as pd

cc23_iucr_codes = pd.read_sql_query("SELECT * FROM cc23_iucr_codes", connection, chunksize = 150)
cc23_fbi_nibrs_categories = pd.read_sql_query("SELECT * FROM cc23_fbi_nibrs_categories", connection, chunksize = 150)
cc23_fbi_nibrs_offense_categories = pd.read_sql_query("SELECT * FROM cc23_fbi_nibrs_offense_categories", connection, chunksize = 150)
cc23_iucr_codes_primary_descriptions = pd.read_sql_query("SELECT * FROM cc23_iucr_codes_primary_descriptions", connection, chunksize = 150)
cc23_iucr_codes_secondary_descriptions = pd.read_sql_query("SELECT * FROM cc23_iucr_codes_secondary_descriptions", connection, chunksize = 150)
cc23_nibrs_crimes_against = pd.read_sql_query("SELECT * FROM cc23_nibrs_crimes_against", connection, chunksize = 150)
cc23_nibrs_fbicode_offenses = pd.read_sql_query("SELECT * FROM cc23_nibrs_fbicode_offenses", connection, chunksize = 150)
cc23_nibrs_offenses_crimes_aginst = pd.read_sql_query("SELECT * FROM cc23_nibrs_offenses_crimes_aginst", connection, chunksize = 150)


# In[8]:


cc23_iucr_codes = pd.concat(list(cc23_iucr_codes))
cc23_fbi_nibrs_categories = pd.concat(list(cc23_fbi_nibrs_categories))
cc23_fbi_nibrs_offense_categories = pd.concat(list(cc23_fbi_nibrs_offense_categories))
cc23_iucr_codes_primary_descriptions = pd.concat(list(cc23_iucr_codes_primary_descriptions))
cc23_iucr_codes_secondary_descriptions = pd.concat(list(cc23_iucr_codes_secondary_descriptions))
cc23_nibrs_crimes_against = pd.concat(list(cc23_nibrs_crimes_against))
cc23_nibrs_fbicode_offenses = pd.concat(list(cc23_nibrs_fbicode_offenses))
cc23_nibrs_offenses_crimes_aginst = pd.concat(list(cc23_nibrs_offenses_crimes_aginst))


# In[9]:


connection.close()


# In[10]:


import getpass
mypasswd = getpass.getpass()
username = 'bat5h8'
host = 'pgsql.dsa.lan'
database = 'dsa_student'

from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine

# SQLAlchemy Connection Parameters
postgres_db = {'drivername': 'postgres',
               'username': username,
               'password': mypasswd,
               'host': host,
               'database' :database}
engine = create_engine(URL(**postgres_db), echo=False)


# In[ ]:


cc23_iucr_codes.to_sql('cc23_iucr_codes',
                       engine,
                       schema = username,
                       if_exists = 'append',
                       index = False,
                       chunksize = 50)


# In[14]:


cc23_iucr_codes_primary_descriptions.to_sql('cc23_iucr_codes_primary_descriptions',
                                            engine,
                                            schema = username,
                                            if_exists = 'append',
                                            index = False,
                                            chunksize = 50)


# In[15]:


cc23_iucr_codes_secondary_descriptions.to_sql('cc23_iucr_codes_secondary_descriptions',
                                            engine,
                                            schema = username,
                                            if_exists = 'append',
                                            index = False,
                                            chunksize = 50)


# In[ ]:


cc23_fbi_nibrs_categories.to_sql('cc23_fbi_nibrs_categories',
                                 engine,
                                 schema = username,
                                 if_exists = 'append',
                                 index = False,
                                 chunksize = 50)


# In[ ]:


cc23_fbi_nibrs_offense_categories.to_sql('cc23_fbi_nibrs_offense_categories',
                                 engine,
                                 schema = username,
                                 if_exists = 'append',
                                 index = False,
                                 chunksize = 50)


# In[ ]:


cc23_nibrs_fbicode_offenses.to_sql('cc23_nibrs_fbicode_offenses',
                                 engine,
                                 schema = username,
                                 if_exists = 'append',
                                 index = False,
                                 chunksize = 50)


# In[ ]:


cc23_nibrs_crimes_against = cc23_nibrs_crimes_against.rename(columns={'nibrs_crime_against': 'nibrs_crimes_against'})


# In[11]:


# reloading based on corrected column name
cc23_nibrs_crimes_against.to_sql('cc23_nibrs_crimes_against',
                                 engine,
                                 schema = username,
                                 if_exists = 'append',
                                 index = False,
                                 chunksize = 50)


# In[ ]:


cc23_nibrs_offenses_crimes_aginst = cc23_nibrs_offenses_crimes_aginst.rename(columns={'nibrs_crime_against': 'nibrs_crimes_against'})


# In[12]:


# relaoding based on corrected column name
cc23_nibrs_offenses_crimes_aginst.to_sql('cc23_nibrs_offenses_crimes_aginst',
                                         engine,
                                         schema = username,
                                         if_exists = 'append',
                                         index = False,
                                         chunksize = 50)


# In[ ]:


for chunk in pd.read_sql_query("SELECT * FROM cc23_cases", connection, chunksize = 1000):
    chunk["arrest"] = chunk["arrest"].astype(bool)
    chunk["domestic"] = chunk["domestic"].astype(bool)
    chunk.to_sql(
    'cc23_cases',
    engine,
    index = False,
    if_exists = 'append'
    )


# In[ ]:


for chunk in pd.read_sql_query("SELECT * FROM cc23_case_location", connection, chunksize = 10000):
    chunk.to_sql(
    'cc23_case_location',
    engine,
    index = False,
    if_exists = 'append'
    )

I ran the above insert table statements, but had originally turned on the Echos, so in an effort to save memory when reloading the page, I cleared the outputs.  See the below record counts.
# In[13]:


connection.close()


# # Execute SQL DML commands (using %sql magic or sqlAlchmey) to confirm the table record counts for the destination database tables are consistent with the source database table record counts

# In[12]:


get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:////dsa/data/DSA-7030/cc23_7030.sqlite.db')


# In[14]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_cases;')


# In[15]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_case_location;')


# In[16]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_iucr_codes;')


# In[17]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_iucr_codes_primary_descriptions;')


# In[18]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_iucr_codes_secondary_descriptions;')


# In[19]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_fbi_nibrs_categories;')


# In[20]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_fbi_nibrs_offense_categories;')


# In[21]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_nibrs_crimes_against;')


# In[22]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_nibrs_fbicode_offenses;')


# In[23]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_nibrs_offenses_crimes_aginst;')


# In[1]:


import getpass
user = "bat5h8"
passwrd = getpass.getpass()
database = "dsa_student"


# In[2]:


get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'postgres://user:passwrd@pgsql.dsa.lan/database')


# In[3]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_cases;')


# In[4]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_case_location;')


# In[5]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_iucr_codes;')


# In[6]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_iucr_codes_primary_descriptions')


# In[7]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_iucr_codes_secondary_descriptions')


# In[8]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_iucr_codes_secondary_descriptions;')


# In[9]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_fbi_nibrs_categories;')


# In[10]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_fbi_nibrs_offense_categories;')


# In[11]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_nibrs_crimes_against;')


# In[12]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_nibrs_fbicode_offenses;')


# In[13]:


get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM cc23_nibrs_offenses_crimes_aginst;')


# ## This is the end of Part 1 of the Final Project 
# ### Part 2 will be deployed in Module 8.

# # Save your notebook, then `File > Close and Halt`
