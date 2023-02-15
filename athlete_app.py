# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 00:49:47 2023

@author: imanj
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide")

plt.rcParams['figure.figsize'] = [12, 8]

dir_list = os.listdir()
athlete = pd.read_csv('athlete_events.csv')
country_data = pd.read_csv('noc_regions.csv')

#Since Singapore's new NOC code for sporting event is SGP instead of SIN, I am replacing it in the dataset

updated_NOC = country_data['NOC'].replace('SIN', 'SGP')

country_data['NOC'] = updated_NOC

#replacing notes with regions

country_data['notes'] = country_data['notes'].fillna(0)

def fix_region(notes):

    country_data['region'] = np.where(country_data['notes'] == 0, country_data['region'], country_data['notes'])

country_data['region'].apply(fix_region)

#merging datasets

data_merged = pd.merge(athlete, country_data, how="outer", on=['NOC'])

#checking data for missing values 

missing_values = data_merged.isna().sum()

# Columns Age, Height, Weight and Medal has missing values

#Attempting to Fill Data with Interpolation or Average
 
#Checking Column Age for Interpolate

interpolate_age = data_merged['Age'].interpolate()

interpolate_age_mean = interpolate_age.mean()

interpolate_age_var = interpolate_age.var()  

#Checking Column Age for Avg

avg_age = data_merged['Age'].fillna(data_merged['Age'].mean())

avg_age_mean = avg_age.mean()

avg_age_var = avg_age.var() 

#Since the mean and variance of Average Age avg_age are closer as compared to the ones for interpolate_age, we will use the mean to fill the missing data

data_merged['Updated_Age'] = avg_age

#Checking Column Height for Interpolate

interpolate_height = data_merged['Height'].interpolate()

interpolate_height_mean = interpolate_height.mean()

interpolate_height_var = interpolate_height.var()  

#Checking Column Height for Avg

avg_height = data_merged['Height'].fillna(data_merged['Height'].mean())

avg_height_mean = avg_height.mean()

avg_height_var = avg_height.var() 

#Since the mean and variance of interpolation are closer as compared to the ones for average, we will use the interpolation to fill the missing data

data_merged['Updated_Height'] = interpolate_height

#Checking Column Weight for Interpolate

interpolate_weight = data_merged['Weight'].interpolate()

interpolate_weight_mean = interpolate_weight.mean()

interpolate_weight_var = interpolate_weight.var()  

#Checking Column Height for Avg

avg_weight = data_merged['Weight'].fillna(data_merged['Weight'].mean())

avg_weight_mean = avg_weight.mean()

avg_weight_var = avg_weight.var() 

#Since the mean and variance of average weight are closer as compared to the ones for interpolation, we will use the interpolation to fill the missing data

data_merged['Updated_Weight'] = avg_weight

#replacing the missing values in column Medal by replacing NAN with 0

fix_medal = data_merged['Medal'].fillna('Others')

data_merged['Updated_Medal'] = fix_medal


#dropping redundant columns

fixed_data = data_merged.drop(columns=['Age', 'Height', 'Weight', 'Medal', 'notes'])

missing_values_fixed_data = fixed_data.isna().sum()

#No missing values in the cleaned data

# NO data types need changing in the data set.

#Checking data for duplicates

duplicate_check = fixed_data.duplicated().sum()

#dropping duplicates from the data

final_data = fixed_data.drop_duplicates()

#Setting a Header and Subheader

st.header('Olympic History Dashboard')

st.subheader('Created by Iman Javaid')

st.text('Historical Dataset on the Modern Olympic Games, including all the Games from Athens 1896 to Rio 2016')

st.markdown('\n')
st.markdown('\n')

#Creating a filter for country
country = st.selectbox('Select Country', sorted(final_data['region'].unique()))

country_cond = final_data['region'] == country 

country_subset = final_data[final_data['region'] == country] 

st.markdown('\n')

#Total Participants Metric

#total_participants =  final_data.groupby(country_cond)['ID'].count().min()

total_participants = len(final_data[(country_cond)])

#Creating Gold Metric

gold_condition = final_data['Updated_Medal'] == 'Gold'
gold_medal = len(final_data[(country_cond) & (gold_condition)])

#gold_subset = country_subset[country_subset['Medal'] == 'Gold']['ID'].count()

#Creating Silver Metric

silver_condition = final_data['Updated_Medal'] == 'Silver'
silver_medal = len(final_data[(country_cond) & (silver_condition)])

#Creating Bronze Metric

bronze_condition = final_data['Updated_Medal'] == 'Bronze'
bronze_medal = len(final_data[(country_cond) & (bronze_condition)])

final_data.to_csv('athlete-cleaned.csv')

with st.container():

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric('Number of Participation Total', total_participants)
    metric2.metric('Number of Gold Medals', gold_medal)
    metric3.metric('Number of Silver Medals', silver_medal)
    metric4.metric('Number of Bronze Medals',bronze_medal)

st.set_option('deprecation.showPyplotGlobalUse', False)

sns.set(style='whitegrid')

st.markdown('\n')
st.markdown('\n')

with st.container():

    line, hbar, table = st.columns(3)
    
# Line Chart for Medal Over the Years
    
    line.subheader('Medals Won Over the Years')
    grouped_data = final_data.groupby(['Year', 'region', 'Updated_Medal']).size().reset_index(name='Medals won in Year')
    
    pivoted_data = grouped_data.pivot_table(index=['Year', 'region'], columns='Updated_Medal', values='Medals won in Year', fill_value=0).reset_index()
    pivoted_data.columns.name = None
    pivoted_data = pivoted_data.rename(columns={'Gold': 'Gold Medals', 'Silver': 'Silver Medals', 'Bronze': 'Bronze Medals'})
    pivoted_data = pivoted_data[['Year', 'region', 'Gold Medals', 'Silver Medals', 'Bronze Medals']]
    
    line_data = pivoted_data[pivoted_data['region'] == country]
    
    if not line_data.empty:
        sns.lineplot(x="Year", y="Gold Medals", data=line_data, label="Gold", color="#E7E000")
        sns.lineplot(x="Year", y="Silver Medals", data=line_data, label="Silver", color="#2A11A0")
        sns.lineplot(x="Year", y="Bronze Medals", data=line_data, label="Bronze", color="#B404F3")
    plt.title('Medals won by '+country+' over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Medals')
    line.pyplot()

# Horizontal Bar Chart of Medals One by athletes 
    hbar.subheader('Top 5 Athletes')
    
    medal_subset = country_subset[country_subset['Updated_Medal'] != 'Others']
    
    athlete_medal = medal_subset.groupby(['ID', 'Updated_Medal', 'Name'])['ID'].count().sort_values(ascending = False).head(5).reset_index(name='Medals')
    #athlete_medal_null = pd.DataFrame(0,0)
    
    if not athlete_medal.empty:
        sns.barplot(x='Medals', y='Name', data=athlete_medal, orient='h', palette = "flare")
    
    plt.title('Top 5 Athletes With Most Medals Won')
    plt.xlabel('No of Medals')
    plt.ylabel('Athletes')
#    plt.yticks(range(len(athlete_medal['Name'])), athlete_medal['Name'])
    hbar.pyplot()
  
    
# Highlighted Table for Medals Won in Each Sport 
    
    table.subheader('Most Medals Won - Sports Wise')
    
    sports_medal = medal_subset.groupby(['Sport'])['Updated_Medal'].count().sort_values(ascending = False).head(5).reset_index(name='Medals')
   
    if not sports_medal.empty: 
        table.dataframe(sports_medal.style.highlight_max(subset='Medals', color = '#189E49'))
        
    if sports_medal.empty:
        fake = pd.DataFrame(columns= ['Sports','Medals'])
        
        table.dataframe(fake)

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')


with st.container():
    hist, pie, vbar = st.columns(3)

# Histogram - Number of Medals Over Age
    hist.subheader('Number of Medals Over Age')
    
    age_medal = medal_subset.groupby(['Updated_Age'])['Updated_Medal'].count().reset_index(name='Medals')
    
    plt.title('Number of Medals Over Age')
    plt.hist(age_medal['Updated_Age'], bins=10, color=['#E77770'])
    
    hist.pyplot()
    
#Pie Chart - Number of Medals of Gender
    pie.subheader('Medals Won by Each Gender (%)')
    
    gender_medal = medal_subset.groupby(['Sex'])['Updated_Medal'].count().reset_index(name='Medals')
#    plt.title('Number of Medals Won Over Gender')
    
    if not gender_medal.empty: 
        gender_colors = {'M': '#ADD8E6', 'F': 'pink'}
        plt.pie(gender_medal['Medals'], labels=gender_medal['Sex'], autopct='%1.1f%%', colors=[gender_colors[gender] for gender in gender_medal['Sex']])
    
    pie.pyplot()
    
# Vertical Bar 
    vbar.subheader('Season Wise Comparison')
    
    season_medal = medal_subset.groupby(['Season'])['Updated_Medal'].count().reset_index(name='Medals')
    
    if not gender_medal.empty:
        season_colors = ['#02dbf0' if season == 'Winter' else '#FFB55B' if season == 'Summer' else 'red' for season in season_medal['Season']]
        sns.barplot(x='Season', y='Medals', data=season_medal, palette=season_colors)
    
    plt.title('Number of Medals Won Season Wise')
    plt.xlabel('Olympic Season')
    plt.ylabel('No of Medals')
    
    vbar.pyplot()
