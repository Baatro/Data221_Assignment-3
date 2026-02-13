import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('crime1.csv')
# isolate the 'Violent Crime' column
violent_crime = data['ViolentCrimesPerPop']

# create histogram
plt.hist(violent_crime)
plt.title('Histogram of Violent Crimes Per Population')
plt.xlabel('ViolentCrimesPerPop')
plt.ylabel('Frequency')
plt.show()

# Box plot
plt.boxplot(violent_crime)
plt.title('Box Plot of Violent Crimes Per Population')
plt.ylabel('ViolentCrimesPerPop Value')
plt.xlabel('Violent Crimes Per Population')
plt.show()

# The histogram shows that the distribution of violent crime rates is right-skewed, with a majority of values clustered towards the lower end (0.02 to 0.4) and a long tail extending towards higher values. 
# This indicates that while most areas have lower violent crime rates, there are some areas with significantly higher rates, which contributes to the right skewness of the distribution.

# The box plot shows that the median is around 0.39(since median is closer to bottom it is right skewed), 
# and there are more outliers above the upper quartile.
