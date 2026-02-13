import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load dataset
kidney_data = pd.read_csv("kidney_disease.csv")

# Clean missing values and convert '?' to NaN 
# Copilot suggestion: kidney_data = kidney_data.replace("?", pd.NA)
kidney_data = kidney_data.replace("?", pd.NA)
kidney_data = pd.get_dummies(kidney_data)
kidney_data = kidney_data.dropna()

X = kidney_data.drop(columns=["classification_notckd", "classification_ckd"])
y = kidney_data["classification_ckd"]   # 1 = ckd, 0 = notckd

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_predict = knn.predict(X_test)

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

# Metrics
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Precision:", precision_score(y_test, y_predict))
print("Recall:", recall_score(y_test, y_predict))
print("F1-score:", f1_score(y_test, y_predict))

#True Positive, True Negative, False Positive, and False Negative
# in the context of kidney disease classification
# True Positive : The model correctly predicts a patient has kidney disease.
# True Negative : The model correctly predicts a patient does not have kidney disease.
# False Positive : The model incorrectly predicts a patient has kidney disease when they do not (Type I error).
# False Negative : The model incorrectly predicts a patient does not have kidney disease when they actually do (Type II error).

# Accuracy isnt reliable, because in cases of imbalanced datasets where one class is much more frequent than the other.
#(Sensitivity) is most important in this case because it measures the proportion of actual positives that are correctly identified. Missing a kidney disease case (false negative) could have severe consequences for the patient.
