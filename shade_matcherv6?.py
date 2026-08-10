import pandas as pd
import joblib
import os

from shade__similarity import find_similar_shades


# loads previously trained model
model = joblib.load("model.pkl")


# dictionaries for user input

brand_map = {
    "huda beauty": 0,
    "fenty beauty": 1,
    "mac": 2
}


brand_names = {
    0: "Huda Beauty",
    1: "Fenty Beauty",
    2: "MAC"
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


# example prediction

example = pd.DataFrame(
    [[1, 4, 2]],
    columns=[
        "BrandCode",
        "DepthScore",
        "UndertoneScore"
    ]
)

prediction = model.predict(example)

print("Example Prediction:", prediction[0])


# user prediction

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
        [[
            brand_code,
            depth_score,
            undertone_score
        ]],
        columns=[
            "BrandCode",
            "DepthScore",
            "UndertoneScore"
        ]
    )


    # predict foundation shade

    prediction = model.predict(user_data)

    recommended_shade = prediction[0]

    print(
        f"\nRecommended Shade: {recommended_shade}"
    )


    # find similar shades

    actual_brand = brand_names[brand_code]

    similar_shades = find_similar_shades(
        actual_brand,
        recommended_shade
    )


    print("\nClosest Shades:")


    if similar_shades is not None:

        for _, row in similar_shades.iterrows():

            print(
                f"- {row['Brand']} {row['Shade']} "
                f"({row['Depth']}, {row['Undertone']}) "
                f"| Score: "
                f"{row['SimilarityScore']:.2f}"
            )


    else:

        print("No similar shades found.")


    # ask for feedback

    correct = input(
        "\nWas this recommendation correct? (y/n): "
    ).strip().lower()


    # if recommendation was incorrect

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
                "Shade",
                "Feedback Type"
            ]
        )


        file_exists = os.path.exists(
            "user_feedback.csv"
        )


        feedback.to_csv(
            "user_feedback.csv",
            mode="a",
            header=not file_exists,
            index=False
        )


        print("\nFeedback saved!")


    # if recommendation was correct

    elif correct == "y":

        feedback = pd.DataFrame(
            [[
                brand_code,
                depth_score,
                undertone_score,
                recommended_shade,
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


        file_exists = os.path.exists(
            "user_feedback.csv"
        )


        feedback.to_csv(
            "user_feedback.csv",
            mode="a",
            header=not file_exists,
            index=False
        )


        print("\nConfirmation saved!")


    # anything other than y or n

    else:

        print(
            "\nInvalid response. Please enter y or n."
        )


except KeyError:

    print(
        "\nInvalid brand, depth, or undertone entered."
    )
