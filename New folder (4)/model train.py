import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
data = pd.read_csv("churn_data.csv")

X = data[['Age', 'Balance', 'EstimatedSalary', 'IsActiveMember']]
y = data['Exited']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as churn_model.pkl")
