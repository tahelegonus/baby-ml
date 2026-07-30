needs to be debugged!!! 

import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df = pd.read_csv("foundation.csv")



print(df.head())


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



X = df[["BrandCode", "DepthScore", "UndertoneScore"]]

y = df["Shade"]

model = DecisionTreeClassifier()

model.fit(X,y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)
model.fit(X_train, y_train)

prediction = model.predict(
    pd.DataFrame(
        [[1,4,2]],
        columns=["BrandCode", "DepthScore", "UndertoneScore"]
    )
)



print(prediction)

brand = int (input("Brand: "))
depth = int(input("Depth: "))
undertone = int(input("Undertone: "))

brand = brand_map[brand]
depth = depth_map[depth]
undertone = undertone_map[undertone]

try:
    brand = brand_map[brand]
except:
    print("Brand not found")

    

prediction = model.predict(
    [[brand, depth, undertone]]
)

print("Recommended shade:", prediction[0])

accuracy = accuracy_score(
    y_test,
    prediction
)

print(accuracy)
