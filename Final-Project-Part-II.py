#!/usr/bin/env python
# coding: utf-8

# # Final Project Part - II DUE: 11:59pm Friday Dec. 15, 2023 
# 
# ## (No late submissions accepted)
# 
# ### In this part II, you will be performing different analytics using the related "cc23_" tables and data you created in your postgres SSO dsa_student database in Part I.
# 
# You will be designing and executing a variety of queries on the Chicago crime database tables - hints included. 
# 
# It is your choice how you connect to your dsa_student SSO database and successfully implement the queries for answering each question.
# 

# In[1]:


#CONNETION HERE
get_ipython().run_line_magic('load_ext', 'sql')
import pandas as pd
import getpass
password = getpass.getpass()
#CONNECTION STRING HERE
get_ipython().run_line_magic('sql', 'postgres://bat5h8:password@pgsql.dsa.lan/dsa_student')

del password


# ### 1- Construct a query to retrieve a count of the primary descriptions of case incidents for all years in descending order. 
# <span style="font-size:7px"><b>Hint:</b> 35 rows affected -- data[PUBLIC PEACE VIOLATION=52825]</span>

# In[2]:


get_ipython().run_cell_magic('sql', '', 'SELECT p.iucr_primary_desc, COUNT(*)\nFROM cc23_iucr_codes_primary_descriptions p\nJOIN cc23_iucr_codes i USING(iucr_code)\nJOIN cc23_cases c USING(iucr_code)\nGROUP BY p.iucr_primary_desc\nORDER BY COUNT(*) DESC;')


# ### 2- Construct a query to retrieve a count of the primary descriptions of case incidents for all years and arrest is TRUE in descending order. 
# <span style="font-size:7px"><b>Hint:</b> 35 rows affected -- data[ARSON=1637]</span>

# In[3]:


get_ipython().run_cell_magic('sql', '', "SELECT p.iucr_primary_desc, COUNT(*)\nFROM cc23_iucr_codes_primary_descriptions p\nJOIN cc23_iucr_codes i USING(iucr_code)\nJOIN cc23_cases c USING(iucr_code)\nWHERE c.arrest = 'true'\nGROUP BY p.iucr_primary_desc\nORDER BY COUNT(*) DESC;")


# ### 3- Construct a query to retrieve the count of case incidents, count and percent (rounded to 3 decimal places) of arrests for each year;  order descending by arrested percent.
# <span style="font-size:7px"><b>Hint:</b> 23 rows affected -- data[2015=26.449%] -- parse dates, uses "case when"</span>

# In[4]:


get_ipython().run_cell_magic('sql', '', "SELECT \nEXTRACT(YEAR FROM incident_date) as year,\nCOUNT(*) as total_incidents,\nSUM(CASE WHEN arrest = 'true' THEN 1 ELSE 0 END) as arrest_count,\nROUND((SUM(CASE WHEN arrest = 'true' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 3) as arrest_percent\nFROM cc23_cases\nGROUP BY year\nORDER BY arrest_percent DESC;")


# ### 4- Construct a query to retrieve the list of iucr codes and index code with their matching primary and secondary descriptions that do not appear as an iucr code for the cases.
# <span style="font-size:7px"><b>Hint:</b> 15 rows affected -- data[1694, N, GAMBLING, POLICY/OFFICE]</span>

# In[5]:


get_ipython().run_cell_magic('sql', '', 'SELECT i.iucr_code, i.iucr_index_code, p.iucr_primary_desc, s.iucr_secondary_desc\nFROM cc23_iucr_codes i\nJOIN cc23_iucr_codes_primary_descriptions p USING(iucr_code)\nJOIN cc23_iucr_codes_secondary_descriptions s USING(iucr_code)\nWHERE i.iucr_code\nNOT IN (\n    SELECT DISTINCT iucr_code\n    FROM cc23_cases\n    WHERE iucr_code IS NOT NULL\n);')


# ### 5- For each year, which month is ranked #1 as having the greatest number of crime incidents with a primary crime description containing the term "NARCOTICS" or secondary crime desciption containing the phrase "GUN"? Display the year, month, of the cases incident date, the primary and secondary crime description, the count of incidents and the ranking number. Order by year in descending order.
# 
# <span style="font-size:7px"><b>Hint:</b>23 rows affected -- data[2012-2, incidents=1942] ranking# should all be 1 -- involves parsing dates, a nested query with a window function and groups</span>

# In[6]:


get_ipython().run_cell_magic('sql', '', "SELECT\nyear,\nmonth,\nprimary_desc,\nsecondary_desc,\nincidents,\nrank\nFROM (\nSELECT\nEXTRACT(YEAR FROM c.incident_date) as year,\nEXTRACT(MONTH FROM c.incident_date) as month,\np.iucr_primary_desc as primary_desc,\ns.iucr_secondary_desc as secondary_desc,\nCOUNT(*) as incidents,\nrank() OVER (PARTITION BY EXTRACT(YEAR FROM c.incident_date) ORDER BY COUNT(*) DESC)\nFROM cc23_cases c\nJOIN cc23_iucr_codes i USING (iucr_code)\nJOIN cc23_iucr_codes_primary_descriptions p USING (iucr_code)\nJOIN cc23_iucr_codes_secondary_descriptions s USING (iucr_code)\nWHERE p.iucr_primary_desc = 'NARCOTICS' OR s.iucr_secondary_desc = 'GUN'\nGROUP BY primary_desc, secondary_desc, year, month\n) rank_data\nWHERE\nrank = 1\nORDER BY year DESC;")


# ### 6 What is the average difference (in days, expressed as an integer) between the updated and incident dates for case arrests and primary and secondary crime descriptions ordered by the average difference in days.
# 
# <span style="font-size:7px"><b>Hint:</b>HINT: 388 rows affected -- output first row data[INTIMIDATION,AGGRAVATED INTIMIDATION,True,19] -- Involves Aggregate groups, parse date, calculate date difference</span>

# In[7]:


get_ipython().run_cell_magic('sql', '', "SELECT \np.iucr_primary_desc as pri_desc, \ns.iucr_secondary_desc as sec_desc, \nFLOOR(AVG(c.updated_on::date - c.incident_date::date)) as difference\nFROM cc23_cases c\nJOIN cc23_iucr_codes i USING(iucr_code)\nJOIN cc23_iucr_codes_primary_descriptions p USING (iucr_code)\nJOIN cc23_iucr_codes_secondary_descriptions s USING (iucr_code)\nWHERE arrest = 'true'\nGROUP BY pri_desc, sec_desc\nORDER BY difference;")


# ### 7 -- Create a query that will retrieve two time series for crime count and arrest count per year from the database. Visualize these two time series in a single plot. 

# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import psycopg2
import sqlalchemy
import getpass

user = "bat5h8"
passwrd = getpass.getpass()
engine = sqlalchemy.create_engine('postgresql://{}:{}@pgsql.dsa.lan/dsa_student'.format(user, passwrd))
connection = engine.connect()

del passwrd

query = """
    SELECT
        EXTRACT(YEAR FROM incident_date) AS year,
        COUNT(*) AS crime_count,
        SUM(CASE WHEN arrest = 'true' THEN 1 ELSE 0 END) as arrest_count
    FROM
        cc23_cases
    GROUP BY year
    ORDER BY year;
"""

result = pd.read_sql_query(query, connection)

plt.figure(figsize = (10,6))
plt.plot(result['year'], result['crime_count'], label = 'Crime Count')
plt.plot(result['year'], result['arrest_count'], label = 'Arrest Count')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Crime and Arrest Counts per Year')
plt.legend()
plt.show()


# In[32]:


connection.close()


# <hr style="border-top: 5px solid green;" />
# 
# ## BONUS QUESTIONS - You have the option to work through items 8 and 9 to receive extra-credit points - but extra-credit points will only be awarded after items 1-7 have been completed.  No bonus if items 1-7 are not faithfully attemtped.
# 
# <hr style="border-top: 5px solid green;" />

# ### 8-9 BONUS (5pts each) -- Create up to 2 novel and useful queries that could be potentially used for policing planning, policy making, citizen awareness, etc. of your choosing.
# 
# Queries **(added to the "Query Here" cells)** should provide some significant analytic value and insight into the Chicago crime data based on a focused domain question of your choosing. Use your SQL skill-set beyond simple SELECT-FROM-WHERE and use multiple tables. Advanced analytic solutions would include GROUP BY/HAVING, Nested Queries, Aggregation Operators, Window Functions, etc.
# 
# Each query should have documentation to explain what this query is attempting to achieve and how it is meaningful and useful for analytic purposes and insight. Add your explanation to the markdown cells below labeled **Documentation/Explanation Here**.

# ### 8 Documentation/Explanation Here
# 
# The query below pulls data from each year and counts the crime based on primary description.  It will then go and graph the trends over the years to see the changes each year.  This would be helpful to see how policing policies have affected crime rate for each crime and then identify potential crime types that could be reduced by potential policies.

# In[1]:


#8 Query Here
# Group by crime and year, then count the number of crimes for that year.  Then graph the trends in a line graph.
# Similar construction to question 7.
# possibly use facet wrap?

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import psycopg2
import sqlalchemy
import getpass

user = "bat5h8"
passwrd = getpass.getpass()
engine = sqlalchemy.create_engine('postgresql://{}:{}@pgsql.dsa.lan/dsa_student'.format(user, passwrd))
connection = engine.connect()

del passwrd

query = """
    SELECT
        EXTRACT(YEAR FROM incident_date) AS year,
        p.iucr_primary_desc as crime,
        COUNT(*) AS crime_count
    FROM
        cc23_cases
    JOIN
        cc23_iucr_codes_primary_descriptions p USING (iucr_code)
    GROUP BY year, crime
    ORDER BY year;
"""

result = pd.read_sql_query(query, connection)
result


# In[2]:


categories = result['crime'].unique()


# In[8]:


fig = plt.figure(figsize=(20, 40))
gs = fig.add_gridspec(len(categories)//2 + 1, 2)

for i, category in enumerate(categories):
    subset_df = result[result['crime'] == category]
    row, col = divmod(i, 2)
    ax = fig.add_subplot(gs[row, col])
    ax.plot(subset_df['year'], subset_df['crime_count'], marker='o')
    ax.set_title(category)

fig.tight_layout()

fig.suptitle('Crime Count per Year and Crime', fontsize=16, x = 0.5, y = 1)

plt.show()


# In[9]:


connection.close()


# ### 9 Documentation/Explanation Here
# 
# https://ucr.fbi.gov/crime-in-the-u.s/2018/crime-in-the-u.s.-2018/topic-pages/violent-crime#:~:text=Definition,force%20or%20threat%20of%20force.
# 
# The above describes how the FBI defines violent crime.  The query below filters by year and count the violent crimes committed in each community area as defined by the FBI.  Additionally, it counts the number of arrests and the arrest percentage in each community.  It finally will rank each community and then filter based on the highest crime count for each community area.  This would be useful to identify the communities that have the highest rate of violent crimes, and how the numbers have changed over the years.  It shows where resources can and should be placed in order to reduce the number of violent crimes.  The arrest percentage will show how resources can be used to increase the number of arrests for these violent crimes.

# In[4]:


get_ipython().run_cell_magic('sql', '', "SELECT year, violent_crime_count, arrest_count, arrest_pct, community\nFROM (\n    SELECT\n        EXTRACT(YEAR FROM c.incident_date) as year,\n        COUNT(*) as violent_crime_count,\n        SUM(CASE WHEN arrest = 'true' THEN 1 ELSE 0 END) as arrest_count,\n        ROUND((SUM(CASE WHEN arrest = 'true' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 3) as arrest_pct,\n        l.community_area as community,\n        rank() OVER (PARTITION BY EXTRACT(YEAR FROM c.incident_date) ORDER BY COUNT(*) DESC)\n    FROM\n        cc23_cases c\n    JOIN\n        cc23_nibrs_fbicode_offenses f ON f.nibrs_offense_code = c.nibrs_fbi_offense_code\n    JOIN\n        cc23_case_location l USING (case_number)\n    WHERE\n        LOWER(f.nibrs_offense_name) LIKE 'aggravated assault%' \n            OR LOWER(f.nibrs_offense_name) LIKE 'rape%'\n            OR LOWER(f.nibrs_offense_name) LIKE 'murder and nonnegligent manslaughter%'\n            OR LOWER(f.nibrs_offense_name) LIKE 'robbery%'\n    GROUP BY\n        year, community\n    ORDER BY\n        year) com_data\nWHERE rank = 1\nORDER BY year DESC;")


# # Save your notebook, then `File > Close and Halt`
