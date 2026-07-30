import pandas as pd 

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("foundation.csv")



print(df.head())

df["BrandCode"] = df["Brand"].map({
    "Huda Beauty": 0,
    "Fenty Beauty": 1,
    "MAC": 2
})


X = df[["BrandCode", "DepthScore", "UndertoneScore"]]

y = df["Shade"]

model = DecisionTreeClassifier()

model.fit(X,y)

prediction = model.predict([[1,4,2]])

print(prediction)

brand = int (input("Brand: "))
depth = int(input("Depth: "))
undertone = int(input("Undertone: "))

prediciton = model.predict([[brand, depth, undertone]])

print("Predicted Shade: ", prediction[0])


