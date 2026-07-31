#loads the model, asks the users, reccomends a shade

import pandas as pd 
import joblib

#loads previously trained model
model = joblib.load("model.pkl")

#supervised learning through labelinf 
#load dataset
#df = pd.read_csv("foundation.csv")

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

#ex prediction
example = pd.DataFrame(
    [[1, 4, 2]],
    columns = ["BrandCode", "DepthScore", "UndertoneScore"]
)

prediction = model.predict(example)

print("Example Prediction:", prediction[0])


# example prediction
example = pd.DataFrame(
    [[1, 4, 2]],
    columns=["BrandCode", "DepthScore", "UndertoneScore"]
)

prediction = model.predict(example)

print("Example Prediction:", prediction[0])


#user prediction

print("\nAvailable Brands")
print("- Huda Beauty")
print("- Fenty Beauty")
print("- MAC")

brand = input("\nBrand: ").strip().lower()
depth = input("Depth: ").strip().lower()
undertone = input("Undertone: ").strip().lower()

try:

    brand_code = brand_map[brand]
    depth_score = depth_map[depth]
    undertone_score = undertone_map[undertone]

    user_data = pd.DataFrame(
        [[brand_code, depth_score, undertone_score]],
        columns=["BrandCode", "DepthScore", "UndertoneScore"]
    )

    prediction = model.predict(user_data)

    print(f"\nRecommended Shade: {prediction[0]}")

except KeyError:

    print("\nInvalid brand, depth, or undertone entered.")
