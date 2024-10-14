import pandas as pd


def find_best_ratios(csv_path) -> pd.DataFrame:
    """Finds the best ratios of Pokémon cards based on their prices relative to the price of a booster pack.

    Args:
        csv_path: Path to the CSV file containing the Pokémon card data.

    Returns:
        DataFrame: DataFrame containing the best ratios of Pokémon cards.
    """
    res: dict[str, list] = {
        "item": [],
        "set_name": [],
        "ungraded_price": [],
        "grade_nine_price": [],
        "grade_ten_price": [],
        "ungraded_nine_ratio": [],
        "ungraded_ten_ratio": [],
        "ungraded_pack_ratio": [],
        "grade_nine_pack_ratio": [],
        "grade_ten_pack_ratio": [],
        "ungraded_box_ratio": [],
        "grade_nine_box_ratio": [],
        "grade_ten_box_ratio": [],
        "booster_pack_price": [],
        "booster_box_price": [],
        "date": [],
    }
    df = pd.read_csv(csv_path)
    # Get all unique sets with Booster Pack listed
    sets = df[df["pokemon_item"].str.contains("Booster Pack")]["set"].unique()
    print(f"Sets with Booster Pack: {sets}")
    sets = [s for s in sets if "japanese" not in s.lower()]
    for pokemon_set in sets:
        booster_pack_price = df[
            (df["set"] == pokemon_set)
            & (df["pokemon_item"].str.contains("Booster Pack"))
        ]["ungraded"].values[0]
        try:
            booster_box_price = df[
                (df["set"] == pokemon_set)
                & (df["pokemon_item"].str.contains("Booster Box"))
            ]["ungraded"].values[0]
        except IndexError:
            booster_box_price = None
        print(f"Booster Pack Price: {booster_pack_price}")
        set_items = df[df["set"] == pokemon_set]
        for _, row in set_items.iterrows():
            if (
                "Booster Pack" in row["pokemon_item"]
                or "Booster Box" in row["pokemon_item"]
                or "Elite Trainer Box" in row["pokemon_item"]
                or "Tin" in row["pokemon_item"]
                or "Box" in row["pokemon_item"]
                or "Pack" in row["pokemon_item"]
            ):
                continue
            res["item"].append(row["pokemon_item"])
            res["set_name"].append(row["set"])
            res["ungraded_price"].append(row["ungraded"])
            res["grade_nine_price"].append(row["grade_nine"])
            res["grade_ten_price"].append(row["grade_ten"])
            try:
                res["ungraded_nine_ratio"].append(row["grade_nine"] / row["ungraded"])
            except Exception:
                res["ungraded_nine_ratio"].append(None)
            try:
                res["ungraded_ten_ratio"].append(row["grade_ten"] / row["ungraded"])
            except Exception:
                res["ungraded_ten_ratio"].append(None)
            res["ungraded_pack_ratio"].append(row["ungraded"] / booster_pack_price)
            res["grade_nine_pack_ratio"].append(row["grade_nine"] / booster_pack_price)
            res["grade_ten_pack_ratio"].append(row["grade_ten"] / booster_pack_price)
            res["booster_pack_price"].append(booster_pack_price)
            if booster_box_price:
                res["ungraded_box_ratio"].append(row["ungraded"] / booster_box_price)
                res["grade_nine_box_ratio"].append(
                    row["grade_nine"] / booster_box_price
                )
                res["grade_ten_box_ratio"].append(row["grade_ten"] / booster_box_price)
                res["booster_box_price"].append(booster_box_price)
            else:
                res["ungraded_box_ratio"].append(None)
                res["grade_nine_box_ratio"].append(None)
                res["grade_ten_box_ratio"].append(None)
                res["booster_box_price"].append(None)
            res["date"].append(row["date(D/M/Y)"])
    return pd.DataFrame(res)


if __name__ == "__main__":
    find_best_ratios("pokemon_prices.csv").to_excel("best_ratios.xlsx", index=False)
