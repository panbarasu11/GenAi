
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)

data = pd.read_csv("diabetes.csv")

print(data.head())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

models = ["Logistic Regression", "Random Forest"]
accuracy = [lr_acc, rf_acc]
loss = [1 - lr_acc, 1 - rf_acc]


plt.figure(figsize=(6,4))
plt.plot(models, accuracy, marker='o')
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(models, loss, marker='o')
plt.title("Model Loss")
plt.ylabel("Loss (1 - Accuracy)")
plt.grid(True)
plt.show()