import pandas as pd
import numpy as np

df = pd.read_csv('../data/nyc311_011523-012123_by022023.csv')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings

# filter out the warning messages
warnings.filterwarnings('ignore', '.*')

# Count the number of complaints for each complaint type and borough
complaint_counts = df.groupby(['Complaint Type', 'Borough']).size().reset_index(name='count')

# Create a bar chart showing the counts for each complaint type
top_complaints = complaint_counts.groupby('Complaint Type')['count'].sum().sort_values(ascending=False).head(20).index.tolist()
top_complaint_counts = complaint_counts[complaint_counts['Complaint Type'].isin(top_complaints)]
sns.set(style="whitegrid")
plt.figure(figsize=(7,8))
ax = sns.barplot(x="count", y="Complaint Type", hue="Borough", data=top_complaint_counts, dodge=False, order=top_complaints)
ax.set_title('Top 20 Complaint Counts by Type and Borough')
plt.show()

# Create a pivot table of top 20 complaint types by hour of day
top_complaints = df['Complaint Type'].value_counts().nlargest(20).index.tolist()
df_top_complaints = df.loc[df['Complaint Type'].isin(top_complaints)]
df_top_complaints['hour'] = pd.to_datetime(df_top_complaints['Created Date']).dt.hour
pivot_df = df_top_complaints.pivot_table(index='hour', columns='Complaint Type', values='Unique Key', aggfunc='count')

# Heatmap of top 20 complaint types by hour of day
plt.figure(figsize=(8,8))
sns.heatmap(pivot_df, cmap='coolwarm', linecolor='white', linewidths=1)
plt.title('Top 20 Complaint Types by Hour of Day')
plt.xticks(rotation=45, ha='right')
plt.show()

# Create a new column for "Noise Related" complaints
df['Complaint Group'] = df['Complaint Type'].apply(lambda x: 'Noise Related' if 'Noise' in x else x)

# Count the number of complaints for each complaint type and borough
complaint_counts = df.groupby(['Complaint Group', 'Borough']).size().reset_index(name='count')

# Create a bar chart showing the counts for each complaint type
top_complaints = complaint_counts.groupby('Complaint Group')['count'].sum().sort_values(ascending=False).head(20).index.tolist()
top_complaint_counts = complaint_counts[complaint_counts['Complaint Group'].isin(top_complaints)]
sns.set(style="whitegrid")
plt.figure(figsize=(7,8))
ax = sns.barplot(x="count", y="Complaint Group", hue="Borough", data=top_complaint_counts, dodge=False, order=top_complaints)
ax.set_title('Top 20 Complaint Counts by Type and Borough')
plt.show()

# create new column "noise" which takes value 1 or 0
df['noise'] = np.where(df['Complaint Group'] == 'Noise Related', 1, 0)
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Filter the DataFrame to only include rows where noise == 1
noise_df = df[df['noise'] == 1]

# Count the number of noise complaints by agency
agency_counts = df[df['noise'] == 1]['Agency Name'].value_counts()

# Plot the results as a bar graph
plt.figure(figsize=(7,6))
agency_counts.plot(kind='bar', color=['blue', 'red', 'maroon'])
plt.title('Number of Noise Complaints by Agency')
plt.xlabel('Agency')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.show()

# Group the data by "Agency Name" and "Descriptor", count the number of complaints, and reset the index
grouped_df = noise_df.groupby(['Agency Name', 'Descriptor'])['Unique Key'].count().reset_index()

# Get a list of unique Agency Names
agency_names = grouped_df['Agency Name'].unique()

# Iterate over the Agency Names and print the top 5 descriptors for each
for agency_name in agency_names:
    print(f"Top 5 descriptors of noise complaints for the {agency_name}:")
    agency_df = grouped_df[grouped_df['Agency Name'] == agency_name].sort_values(by='Unique Key', ascending=False)
    top_5_descriptors = agency_df['Descriptor'].head(5).values.tolist()
    print(", ".join(top_5_descriptors))
    print()

# new column "hour", which takes integer (1-23) of hour of reequest
df['hour'] = pd.to_datetime(df['Created Date']).dt.hour

# pibvot table when noise=1, per hour
noise_pivot = df[df["noise"] == 1].pivot_table(index='hour', values='Unique Key', aggfunc='count')

# Plot the pivot table as a line chart
sns.lineplot(x=noise_pivot.index, y=noise_pivot['Unique Key'], color='blue')
plt.title("# of Noise Complaints by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("# of Noise Complaints")
plt.show()

# Create a new column with the day of the week
noise_df['day_of_week'] = pd.to_datetime(noise_df['Created Date']).dt.day_name()

# Convert "day_of_week" column to a categorical data type with order
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
noise_df["day_of_week"] = pd.Categorical(noise_df["day_of_week"], categories=day_order, ordered=True)

# Count the number of noise complaints per day of the week
day_counts = noise_df['day_of_week'].value_counts()

# Create bar plot of noise complaints per day of the week
plt.figure(figsize=(10,6))
sns.barplot(x=day_counts.index, y=day_counts.values, palette='bright')
plt.title("Number of Noise Complaints per Day of the Week", fontsize=16)
plt.xlabel("Day of the Week", fontsize=14)
plt.ylabel("Number of Complaints", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

import numpy as np

# create day_of_week column in original df
df['day_of_week'] = pd.to_datetime(df['Created Date']).dt.day_name()

# split df into weekends or weekdays
weekdays = df[df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])]
weekends = df[df['day_of_week'].isin(['Friday', 'Saturday', 'Sunday'])]

# calculate mean and std for t test
mean_weekdays = np.mean(weekdays['noise'])
std_weekdays = np.std(weekdays['noise'])

mean_weekends = np.mean(weekends['noise'])
std_weekends = np.std(weekends['noise'])

# plot pie chart
fig, ax = plt.subplots()

labels = ['Weekday', 'Weekend']
sizes = [mean_weekdays, mean_weekends]
colors = ['#4FB0C6', '#87C540']

ax.pie(sizes, labels=labels, colors=colors,
       autopct='%1.1f%%', startangle=90, pctdistance=0.75, labeldistance=1.1, textprops={'fontsize': 14})

# set aspect ratio to be equal so that pie is drawn as a circle
ax.axis('equal')  
plt.title('Proportion of Noise Complaints by Day Type', fontsize=16)
plt.show()

from scipy.stats import ttest_ind

# Do two sample t test for diff in weekends vs weekdays
t_stat, p_val = ttest_ind(weekdays['noise'], weekends['noise'])

# printing p value and conclusion
print("P-value:", p_val)

if p_val < 0.05:
    print("The total number of noise complaints is significantly greater on weekends than weekdays.")
else:
    print("There is no significant difference in the frequency of noise complaints between weekdays and weekends.")

# Count of noise complaints by borough
noise_borough = df[df['noise'] == 1].groupby('Borough').size().reset_index(name='count')
noise_borough = noise_borough.sort_values('count', ascending=False)

# Set the color palette
colors = sns.color_palette('bright', len(noise_borough))

# Create the bar plot
plt.figure(figsize=(10,6))
plt.bar(noise_borough['Borough'], noise_borough['count'], color=colors)
plt.title('Number of Noise Complaints Per Borough')
plt.xlabel('Borough')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()

import scipy

# Chi-squared test for noise complaints by borough
noise_counts = noise_df['Borough'].value_counts()
borough_counts = df['Borough'].value_counts()
observed = noise_counts.reindex(borough_counts.index, fill_value=0)
expected = borough_counts * (observed.sum() / borough_counts.sum())
chi2, p, dof, expected = scipy.stats.chi2_contingency([observed, expected])
# Printing results of chi sq test
print(f"Chi-squared test statistic: {chi2}")
print(f"p-value: {p}")

import gmplot
from apikey import api_key #my own api key in seperate file

# update df to only include valid 5 boroughs
df = df[df['Borough'] != 'Unspecified']

# Create a subset of the data for noise complaints only
noise_df = df[df['noise'] == 1]

# Group the dataframe by the "BOROUGH" column
borough_groups = noise_df.groupby('Borough')

# Creating a map centered around NYC with the location points scattered
gmap = gmplot.GoogleMapPlotter.from_geocode('New York City', 10, apikey=api_key)

# Loop through each group and plot the points with a different color for each borough
colors = ['red', 'blue', 'green', 'brown', 'purple'] # define a list of colors for each borough
for i, (borough, group) in enumerate(borough_groups):
    gmap.scatter(group['Latitude'], group['Longitude'], marker=False, size=75, color=colors[i])

# Draw the map
gmap.draw('noise_map.html')

# importing uszipcode to merge
from uszipcode import SearchEngine, SimpleZipcode, ComprehensiveZipcode
import sqlite3

# Removing the .0 and changing to str to match zip code database
noise_df['Incident Zip'] = noise_df['Incident Zip'].astype(str).str[:-2]

# Retrieving database from uszipcode and assigning it to zipc (dataframe)
with sqlite3.connect("/Users/lukesmac/.uszipcode/simple_db.sqlite") as con:
    zipc = pd.read_sql_query("SELECT * from simple_zipcode", con)

# Merging the two dataframes using zip code key, assigning to new dataframe
noise_zip = pd.merge(noise_df, zipc, how='left', left_on='Incident Zip', right_on='zipcode')

# Group data by zip code and count the number of rows for each group
num_complaints = noise_zip.groupby('Incident Zip')['Unique Key'].count()

# Convert the series to a dataframe
df_num_complaints = pd.DataFrame(num_complaints.reset_index())

# new df that contains count of noise
df_num_complaints = df_num_complaints.rename(columns={'Unique Key': 'count'})

# merge uszipcode with this new count df
count_zip = pd.merge(df_num_complaints, zipc, how='left', left_on='Incident Zip', right_on='zipcode')

# create a scatter plot of median household income vs number of noise complaints, with color representing population
plt.scatter(count_zip['median_household_income'], count_zip['count'], c=count_zip['population'], cmap='viridis')

# add axis labels and a title
plt.xlabel('Median Household Income')
plt.ylabel('Number of Noise Complaints')
plt.title('Relationship between Median Household Income and # of Noise Complaints')

# add a colorbar legend
cb = plt.colorbar()
cb.set_label('Population')

# show the plot
plt.show()

# Calculate correlation coefficient
correlation = count_zip['median_household_income'].corr(count_zip['count'])
# printing corr
print('Correlation coefficient:', correlation)