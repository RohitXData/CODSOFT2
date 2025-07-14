# iris_classification.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names.tolist()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model trained successfully!")

# Predict on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.2f}")

# --- User Input ---
print("\n--- Predict your own Iris flower ---")
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)
predicted_class = iris.target_names[prediction[0]]

print(f"\nPredicted Iris species: {predicted_class}")
