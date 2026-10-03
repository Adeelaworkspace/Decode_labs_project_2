# Project 2: Data Classification Using AI

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# 1. Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target


# 2. Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Create KNN Model
model = KNeighborsClassifier(n_neighbors=5)


# 5. Train the model
model.fit(X_train, y_train)


# 6. Make predictions
predictions = model.predict(X_test)


# 7. Check results
accuracy = accuracy_score(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")


print("===== Iris Flower Classification =====")

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion)

print("\nF1 Score:", f1)