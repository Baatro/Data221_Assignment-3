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