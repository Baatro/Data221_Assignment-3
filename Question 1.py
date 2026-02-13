import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('crime1.csv')
# isolate the 'Violent Crime' column
violent_crime = data['ViolentCrimesPerPop']

# Get stats
mean = violent_crime.mean()
median = violent_crime.median()
standard_deviation = violent_crime.std()
min_value = violent_crime.min()
max_value = violent_crime.max()

# Print the results
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {standard_deviation}")
print(f"Minimum: {min_value}")
print(f"Maximum: {max_value}")


# Mean: 0.44119122257053295
# Median: 0.39
# Standard Deviation: 0.2763505847811399
# Minimum: 0.02
# Maximum: 1.0

# Since the mean (0.441) is higher than the median (0.39), it suggests that the distribution of violent crime rates is right-skewed distribution, this means majority of 
# data values are clustered on the left (lower values), with a few extreme, higher values stretching the tail to the right. 


#having extreme values (outliers) would pull the mean up, while the median would be less affected by those outliers bceause mean is the average of all values, while median is the middle value when the data is sorted.