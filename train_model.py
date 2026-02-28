import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib



# Load dataset
data = pd.read_csv("learning_data_balanced.csv")

# Features (Input)
X = data[["watches_videos", "prefers_audio", "likes_practical", "reads_notes"]]

# Target (Output)
y = data["learning_style"]

# Create Decision Tree model
model = DecisionTreeClassifier()

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")