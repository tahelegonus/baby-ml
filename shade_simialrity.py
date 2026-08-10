import math
import pandas as pd


def hex_to_rgb(hex_color):

    hex_color = hex_color.lstrip("#")

    return tuple(
        int(hex_color[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def color_distance(color1, color2):

    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)

    distance = math.sqrt(
        (r1 - r2) ** 2 +
        (g1 - g2) ** 2 +
        (b1 - b2) ** 2
    )

    return distance


def similarity_score(
    color_distance_value,
    target_depth,
    target_undertone,
    shade_depth,
    shade_undertone
):

    depth_difference = abs(
        target_depth - shade_depth
    )

    undertone_difference = abs(
        target_undertone - shade_undertone
    )

    score = (
        (color_distance_value * 0.60) +
        (depth_difference * 10 * 0.25) +
        (undertone_difference * 10 * 0.15)
    )

    return score


def find_similar_shades(brand, shade, number_of_results=5):

    df = pd.read_csv("foundation.csv")

    target = df[
        (df["Brand"] == brand) &
        (df["Shade"].astype(str) == str(shade))
    ]

    if target.empty:
        return None

    target_hex = target.iloc[0]["HexColor"]

    target_depth = target.iloc[0]["DepthScore"]

    target_undertone = target.iloc[0]["UndertoneScore"]

    df["ColorDistance"] = df["HexColor"].apply(
        lambda x: color_distance(target_hex, x)
    )

    df["SimilarityScore"] = df.apply(
        lambda row: similarity_score(
            row["ColorDistance"],
            target_depth,
            target_undertone,
            row["DepthScore"],
            row["UndertoneScore"]
        ),
        axis=1
    )

    similar_shades = df.sort_values(
        "SimilarityScore"
    )

    # remove the exact shade being searched

    similar_shades = similar_shades[
        ~(
            (similar_shades["Brand"] == brand) &
            (similar_shades["Shade"].astype(str) == str(shade))
        )
    ]

    return similar_shades.head(number_of_results)
