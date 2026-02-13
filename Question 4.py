import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load dataset
kidney_data = pd.read_csv("kidney_disease.csv")

# Clean missing values and convert '?' to NaN
kidney_data = kidney_data.replace("?", pd.NA)

# One-hot encode categorical columns
kidney_data = pd.get_dummies(kidney_data)

# Drop remaining missing values
kidney_data = kidney_data.dropna()

# Create feature matrix and label vector
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
