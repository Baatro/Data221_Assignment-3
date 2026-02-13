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
y = kidney_data["classification_ckd"] 

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Train KNN
test_K_values = [1, 3, 5, 7, 9]

results_table = [("K", "Accuracy")]
for k in test_K_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict
    y_predict = knn.predict(X_test)


    # Store into a table (k, accuracy)
    results_table.append((k, accuracy_score(y_test, y_predict)))

# Print
results_df = pd.DataFrame(results_table[1:], columns=results_table[0])
print("Results Table:")
print(results_df.to_string(index=False))

# We can see that as k increases, the accuracy tends to decrease. This is because with a larger k, the model becomes more generalized.
# While smaller k values (like 1 or 3) can capture more local patterns in the data, they are also more sensitive to noise and outliers, which can lead to overfitting.
# On the other hand, larger k values (like 7 or 9) may smooth out the decision boundary too much, causing the model to underfit and miss important patterns in the data.
