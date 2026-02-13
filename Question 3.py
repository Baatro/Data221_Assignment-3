import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
kidney_data = pd.read_csv("kidney_disease.csv")

# Create feature matrix 
X = kidney_data.drop(columns=["classification"])
y = kidney_data["classification"]

# Split into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)


# We should not train and test a model on the same data because the model would memorize
# the training examples instead of learning general patterns. This would make the model appear
# to perform well, but only because it has already seen the answers. The purpose of the testing
# set is to evaluate how well the model performs on data it has never encountered before.
# A separate testing set helps us estimate the model's true predictive ability in real-world use.
# Without a proper train/test split, we would have no reliable way to detect overfitting.
