#model uses this file to save results from shade matcher and store it 

import pandas as pd 
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#supervised learning through labeling 

#load dataset
df = pd.read_csv("foundation.csv")

#convert brand names into numbers
df["BrandCode"] = df["Brand"].map({
    "Huda Beauty": 0,
    "Fenty Beauty": 1,
    "MAC": 2
})

# features
X = df[["BrandCode", "DepthScore", "UndertoneScore"]]

# target
y = df["Shade"]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

#create the model
model = DecisionTreeClassifier(random_state=42)

#train the model
model.fit(X_train, y_train)

# evaluate the model
test_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# dictionaries for user input
brand_map = {
    "huda beauty": 0,
    "fenty beauty": 1,
    "mac": 2
}

depth_map = {
    "very light": 1,
    "light": 2,
    "medium light": 3,
    "medium": 4,
    "medium dark": 5,
    "dark": 6
}

undertone_map = {
    "cool": 0,
    "neutral": 1,
    "warm": 2
}

# features
X = df[["BrandCode", "DepthScore", "UndertoneScore"]]

# target
y = df["Shade"]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

#create the model
model = DecisionTreeClassifier(random_state=42)

#train the model
model.fit(X_train, y_train)

# evaluate the model
test_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)

print(f"Model Accuracy: {accuracy:.2f}")

#save the trained model
joblib.dump(model, "model.pkl")

print("Model saved Succesfully!")
