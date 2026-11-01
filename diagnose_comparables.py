"""Diagnose historical comparable-sale fallback levels without fitting XGBoost.

Run from your SSU-Data-Dash repository root:
    uv run python diagnose_comparables.py

Uses the same December 2025--April 2026 walk-forward months as run.py.
Outputs are written to experiments/06_xgb_historical_sales/diagnostics/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import data
from comparable_sales import ComparableSalesFeatures
from run import VALIDATION_MONTHS


OUT = Path("experiments/06_xgb_historical_sales/diagnostics")
LEVELS = {
    0: "No usable history",
    1: "All historical homes",
    2: "Same house type",
    3: "Same type + area band",
    4: "Same community + type",
    5: "Community + type + area band",
}


def main():
    listings = data.load("listings")
    sold_month = listings["date_sold"].dt.to_period("M")
    results = []

    for month in VALIDATION_MONTHS:
        # Exactly the sales that customValidate() makes available to the model
        # before scoring this month (its training + early-stopping data).
        history = listings.loc[listings["date_sold"] < month.start_time]
        homes = listings.loc[sold_month == month]

        # Pass only permitted property fields to the historical feature generator.
        visible = homes[["ml_num", "date_listed", "community",
                         "house_type_name", "house_area"]].copy()
        builder = ComparableSalesFeatures().fit(history)
        features = builder.transform(visible)

        result = pd.DataFrame({
            "ml_num": visible["ml_num"].to_numpy(),
            "month": str(month),  # validation *sale* month, used to match predictions
            "house_type_name": visible["house_type_name"].to_numpy(),
            "past_level_12m": features["past_level_12m"].to_numpy(),
            "past_count_12m": features["past_count_12m"].to_numpy(),
            "past_ppsf_12m": features["past_ppsf_12m"].to_numpy(),
        })
        results.append(result)
        print(f"Scanned {month}: {len(homes):,} homes")

    all_homes = pd.concat(results, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    all_homes.to_csv(OUT / "comparable_diagnostics.csv", index=False)

    summary = (all_homes.groupby("past_level_12m")
               .agg(homes=("ml_num", "size"),
                    median_comps=("past_count_12m", "median"))
               .reindex(range(6), fill_value=0)
               .reset_index())
    summary["meaning"] = summary["past_level_12m"].map(LEVELS)
    summary["pct_homes"] = 100 * summary["homes"] / len(all_homes)
    summary.to_csv(OUT / "fallback_summary.csv", index=False)

    by_type = (all_homes.pivot_table(index="house_type_name",
                                     columns="past_level_12m", values="ml_num",
                                     aggfunc="count", fill_value=0)
               .reindex(columns=range(6), fill_value=0))
    by_type.columns = [f"level_{c}" for c in by_type.columns]
    by_type["total"] = by_type.sum(axis=1)
    by_type["pct_level_5"] = (100 * by_type["level_5"] / by_type["total"]).round(2)
    by_type.to_csv(OUT / "fallback_by_house_type.csv")

    by_month = pd.crosstab(all_homes["month"], all_homes["past_level_12m"])
    by_month = by_month.reindex(columns=range(6), fill_value=0)
    by_month.columns = [f"level_{c}" for c in by_month.columns]
    by_month["total"] = by_month.sum(axis=1)
    by_month["pct_level_5"] = (100 * by_month["level_5"] / by_month["total"]).round(2)
    by_month.to_csv(OUT / "fallback_by_month.csv")

    print("\n=== How often was each fallback level used? ===")
    print(summary[["past_level_12m", "meaning", "homes", "pct_homes", "median_comps"]]
          .to_string(index=False, formatters={"pct_homes": "{:.1f}%".format}))
    print(f"\nSpecific groups (level 5): {(all_homes['past_level_12m'] == 5).mean():.1%}")
    print(f"Broad fallback (levels 1-3): {all_homes['past_level_12m'].between(1, 3).mean():.1%}")
    print(f"Missing history (level 0): {(all_homes['past_level_12m'] == 0).mean():.1%}")
    print("\n=== Specific-group coverage by property type ===")
    print(by_type[["total", "pct_level_5"]].to_string())

    # Optional: compare saved old/new XGBoost prediction errors within each level.
    old_path = Path("experiments/05_xgb_all_categorical/predictions.csv")
    new_path = Path("experiments/06_xgb_historical_sales/predictions.csv")
    if old_path.exists() and new_path.exists():
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_path)
        keys = ["ml_num", "month", "track"]
        paired = old[keys + ["sold_price", "predicted_price"]].merge(
            new[keys + ["sold_price", "predicted_price"]],
            on=keys, validate="one_to_one", suffixes=("_old", "_new")
        )
        if len(paired) != len(old) or len(paired) != len(new):
            raise ValueError("Saved prediction files do not contain matching rows.")
        if not np.allclose(paired["sold_price_old"], paired["sold_price_new"]):
            raise ValueError("Actual sale prices disagree between the two experiments.")
        paired = paired.merge(all_homes[keys[:2] + ["past_level_12m"]],
                              on=keys[:2], validate="many_to_one")
        if len(paired) != len(old):
            raise ValueError("Comparable diagnostics and predictions do not align.")
        for label in ["old", "new"]:
            paired[f"ape_{label}"] = 100 * (
                (paired[f"predicted_price_{label}"] - paired[f"sold_price_{label}"]).abs()
                / paired[f"sold_price_{label}"]
            )
        errors = (paired.groupby(["track", "past_level_12m"])
                  .agg(homes=("ml_num", "size"),
                       old_mdape=("ape_old", "median"),
                       new_mdape=("ape_new", "median"),
                       old_mape=("ape_old", "mean"),
                       new_mape=("ape_new", "mean")).round(2).reset_index())
        errors["mdape_change_pp"] = (errors["new_mdape"] - errors["old_mdape"]).round(2)
        errors.to_csv(OUT / "error_comparison_by_level.csv", index=False)
        print("\n=== Old vs historical-sales XGBoost (lower error is better) ===")
        print(errors.to_string(index=False))
    else:
        print("\nNo paired error analysis: need both saved prediction files at:")
        print("  ", old_path)
        print("  ", new_path)

    print("\nSaved diagnostic CSVs to:", OUT)


if __name__ == "__main__":
    main()
