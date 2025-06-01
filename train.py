import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data_path = "iris.csv"
df = pd.read_csv(data_path)
X = df.drop(columns="target")
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "weights.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# # Save metrics
# metrics_path = "metrics.csv"
# with open(metrics_path, "w") as f:
#     f.write("accuracy\n")
#     f.write(f"{accuracy:.4f}\n")
