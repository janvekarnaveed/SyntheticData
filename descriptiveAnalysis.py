# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:01:10 2016

@author: naveedjanvekar
"""

import pandas as pd
import csv
from glob import glob
import os
import numpy as np

#Concatenating all the text files into one csv file

#1. Create a list of state abbreviations using file names
path = "namesbystate/" #Path of babynames dataset folder
dfile = open("all_state.csv", 'w')
# write the headers
dfile.write("state,sex,year,name,count\n")
# now glob all the files into a list
txtfiles = glob(path + '*.TXT')
print("There are", len(txtfiles), "textfiles")
for fn in txtfiles:
    # get the year from the filename
    state = os.path.basename(fn).split('.')[0]
    print("State:", state)
    with open(fn) as f:
#        # no need to add things to each line, just copy the whole thing
#        # iterate through each line
        dfile.writelines(f.readlines())

#Reading the concatenated csv file

data = pd.read_csv('all_state.csv',sep=',' )

#Question2 -- Most Popular Names of all time

data2 = data.groupby(by=['name', 'sex'])['count'].sum().reset_index()
data2 = data2.sort_values('count', ascending = False)

data2_male = data2.loc[data2['sex'] == 'M']
data2_female = data2.loc[data2['sex'] == 'F']

print ("The most popular name across either gender is ", data2.head(10))
print ("The most popular name across males is ", data2_male.head(10))
print ("The most popular name across females is", data2_female.head(10))


#The most popular name of all time (Of either gender.) is James
#The most popular male name is James
#The most popular female name is Mary


#Question 3 -- Finding the most ambiguous name in any given year
def dataQuestion3(data, yearFind):
    data_year = data.groupby(by=['name', 'sex', 'year'])['count'].sum().reset_index()
    data_year = data_year.sort_values('count', ascending = False)
    data_year = data_year.loc[data_year['year'] == yearFind]
    
    
    data_year_dup = data_year[data_year.duplicated(['name'],keep=False)]
    data_year_dup = data_year_dup.sort_values(by=['name', 'count'], ascending=[True, True])
    
    
    data_year_dup_male = data_year_dup.loc[data_year_dup['sex'] == 'M']
    data_year_dup_female = data_year_dup.loc[data_year_dup['sex'] == 'F']
    
    data_year_dup_male=data_year_dup_male.rename(columns = {'count':'count_m'})
    data_year_dup_female=data_year_dup_female.rename(columns = {'count':'count_f'})
    
    result = pd.merge(data_year_dup_male, data_year_dup_female, on='name').sort_values(by=['count_f','count_m',], ascending=[False, False])
    
    #COunt diff will calculate the absolute difference between number of male occurences and numbe of female occurences [ count_diff = ABS (%M -%F) ]
    result['count_diff']= (result['count_m']-result['count_f']).abs()
    result_diff =  result.sort_values(by=['count_diff','name'], ascending=[True, True])
    
    print ("The most gender ambiguous name is", result_diff.head(10))

#the most gender ambiguous name in 2013 is Nikita with a total occurences for 47 males and 47 females
#the most gender ambiguous name in 1945 is Maxie with a total occurences for 19 males and 19 females

dataQuestion3(data, 2013)
dataQuestion3(data, 1945)


#Question 4 -- Largest percentage increase and decrease in popularity of names

def dataQuestion4(data, year1, year2):

    data_year = data.groupby(by=['name', 'year'])['count'].sum().reset_index()
    data_year = data_year.sort_values('count', ascending = False)
    
    data_year1 = data_year.loc[data_year['year'] == year1]
    data_year2 = data_year.loc[data_year['year'] == year2]
    
    #Calculating the popularity of a given name in any given year
    data_year1["pop_r_year1"] = (data_year1["count"]/data_year1["count"].sum())*100
    data_year2["pop_r_year2"] = (data_year2["count"]/data_year2["count"].sum())*100
    
    
    #Renaming the count columns to perform merge operation
    data_year1=data_year1.rename(columns = {'count':'count_year1'})
    data_year2=data_year2.rename(columns = {'count':'count_year2'})
    
    # Making sure to compare names which existed in year2 and still exist in year1
    data_year1_year2 = pd.merge(data_year1, data_year2, on='name')
    
    #Calculating the percentage increase in the name
    data_year1_year2["diff"] = data_year1_year2["pop_r_year1"]-data_year1_year2["pop_r_year2"]
    data_year1_year2 = data_year1_year2.sort_values('diff', ascending = False)
    
    
   #Printing the results
    print ("largest percentage increase in popularity since %d" %year2, data_year1_year2.head(20))
    print ("largest percentage decrease in popularity since %d" %year2 , data_year1_year2.tail(20))


dataQuestion4(data, 2013, 1980) #Calling dataQuestion4 function to find percentage increase in name since 1980 to 2013
dataQuestion4(data, 2014, 1980) #Calling dataQuestion4 function to find percentage increase in name since 1980 to 2014

#Question 5 - Identifying names that may have had an even larger increase or decrease in popularity
dataQuestion4(data, 2014, 1910)


#######################################B) Onward to Insights#######################################
#While working with the data, I have noticed that there seems to be many more female names compared to males, so I decided to explore this
#and plot diversity of female vs. male names on the timeline (where diversity is measured by the total number of unique names per gender per year)

#Subsetting babynames data set as separate data for male names and female names
data2_male = data.loc[data['sex'] == 'M']
data2_female = data.loc[data['sex'] == 'F']

#Count of Unique names over the years
data_male = data2_male.groupby(['year'])['name'].apply(lambda x: len(x.unique())).reset_index()
data_female = data2_female.groupby(['year'])['name'].apply(lambda x: len(x.unique())).reset_index()

#Renaming the columns name which has count of unique names over the years to male and female to avoid conflict while merging data_male and data_female
data_male=data_male.rename(columns = {'name':'male'})
data_female=data_female.rename(columns = {'name':'female'})
    
#Insight 1 -- Name Diversity    
data_insight1 = pd.merge(data_male, data_female, on='year')

ax1=data_insight1.plot(x='year',y=['male','female'], marker='.',title = "Male vs Female Name Diversity")
ax1.set_xlabel("Year")
ax1.set_ylabel("Unique Names Used")

#Insight 2 --  Male to Female Diversity ratio
data_insight1["ratioMF"] = data_insight1["male"]/data_insight1["female"]

ax2=data_insight1.plot(x='year',y=['ratioMF'], marker='.',title = " % Ratio of Male vs Female Unique Names per year")
ax2.set_xlabel("Year")
ax2.set_ylabel("% of Males to Females Unique names")

#Insight 3 -- To analyze immigration data with baby names diversity
#External dataset from https://www.dhs.gov/publication/yearbook-immigration-statistics-2013-lawful-permanent-residents
df1 = pd.read_excel('table1_3.xls')

#Cleansing the immigration dataset
df2 = df1.ix[3:]
df3 = df2[:-2]

df4_1 = df3.ix[:,0:2]
df4_1.columns=['year','number']
df4_2 = df3.ix[:,2:4]
df4_2.columns=['year','number']
df4_3 = df3.ix[:,4:6]
df4_3.columns=['year','number']
df4_4 = df3.ix[:,6:8]
df4_4.columns=['year','number']


merge1 = pd.concat([df4_1,df4_2], axis=0)
dfMerged1 = pd.merge(df4_1, df4_2,
                    left_on=['year','number'],right_on=['year','number'],how='outer')
                    
dfMerged2 = pd.merge(df4_3, df4_4,
                    left_on=['year','number'],right_on=['year','number'],how='outer')

#Cleansed immigration dataset                    
df_finalMerge = pd.merge(dfMerged1, dfMerged2,
                         left_on=['year','number'],right_on=['year','number'],how='outer').dropna()

#Plotting the immigration statistics to corraborate with the earlier findings
ax3=df_finalMerge.plot(x='year',y='number', marker='.',title = " Immigration Statistics")
ax3.set_xlabel("Year")
ax3.set_ylabel("Number of Immigrants")



#Insight 4 - Average length of name every year
def compute_average(year):
    # Computes the average length of the names for each year
    x = (data.name[data.year==year])
    y = x.str.len()
    return np.average(y)

# Use two lists to store simultaneously the average length and the year
averages = []
years = []

# Loop over the years 
for year in data.year.unique():
    averages.append(compute_average(year))
    years.append(year)
    
    
dataAvgLen = pd.DataFrame({'year' : years, 'average' : averages })

ax4=dataAvgLen.plot(x='year',y='average', marker='.',title = "Average Length of Names over the years")
ax4.set_xlabel("Year")
ax4.set_ylabel("Average Length of Names")









