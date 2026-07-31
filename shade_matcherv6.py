import pandas as pd 
import joblib
import os 

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

    correct = input(
        "\nWas this recommendation correct? (y/n): "
    ).strip().lower()


    if correct == "n":

        actual_shade = input(
            "What is your correct shade? "
        ).strip()

        feedback = pd.DataFrame(
            [[
                brand_code,
                depth_score,
                undertone_score,
                actual_shade,
                "correction"
            ]],
            columns=[
                "BrandCode",
                "DepthScore",
                "UndertoneScore",
                "Shade"
                "Feddback Type"
            ]
        )


        file_exists = os.path.exists("user_feedback.csv")

        feedback.to_csv(
            "user_feedback.csv",
            mode="a",
            header=not file_exists,
            index=False
        )

        print("\nFeedback saved!")


    elif correct == "y":

        feedback = pd.DataFrame(
            [[
                brand_code,
                depth_score,
                undertone_score,
                prediction[0],
                "confirmed"
            ]],
            columns=[
                "BrandCode",
                "DepthScore",
                "UndertoneScore",
                "Shade",
                "Feedback Type"
            ]
        )


        file_exists = os.path.exists("user_feedback.csv")

        feedback.to_csv(
            "user_feedback.csv",
            mode="a",
            header=not file_exists,
            index=False
        )

        print("\nConfirmation Save!")


    else:

        print("\nInvalid response. Please enter y or n.")


except KeyError:

    print("\nInvalid brand, depth, or undertone entered.")
