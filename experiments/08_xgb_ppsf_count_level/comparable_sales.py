"""Leakage-aware historical comparable-sale features for the SSU Data Dash.

For each target listing month M, look only at sales completed in the 12 FULL
calendar months before M. Never use the target home's sold price or sale date.

This is an independent implementation inspired by Quinton's discussion, not
copied from Quinton's unpublished code.
"""

import numpy as np
import pandas as pd

from features import square_feet


class ComparableSalesFeatures:
    # Fixed area ranges in square feet. A comparison always uses known attributes.
    AREA_BOUNDARIES = [800, 1200, 1800, 2500, 3500]
    MIN_COMPS = 5

    def fit(self, sales):
        """Store completed sales available to this fold (no model is trained here)."""
        history = sales[[
            "date_sold", "date_listed", "community", "house_type_name", "house_area", "sold_price"
        ]].copy()
        history["date_sold"] = pd.to_datetime(history["date_sold"], errors="coerce")
        history["date_listed"] = pd.to_datetime(history["date_listed"], errors="coerce")
        history["sqft"] = square_feet(history["house_area"])
        history["sold_price"] = pd.to_numeric(history["sold_price"], errors="coerce")
        history["price_per_sqft"] = history["sold_price"] / history["sqft"]
        history = history.loc[
            history["date_sold"].notna()
            & history["date_listed"].notna()
            & (history["date_sold"] >= history["date_listed"])
            & np.isfinite(history["price_per_sqft"])
            & (history["sqft"] > 0)
            & (history["sold_price"] > 0)
        ].copy()
        history["area_band"] = self._area_bands(history["sqft"])
        # Pandas string columns make merges robust even when the raw columns
        # have category dtype in another part of the pipeline.
        for col in ("community", "house_type_name"):
            history[col] = history[col].astype("string").fillna("<missing>")
        self.history = history
        return self

    @classmethod
    def _area_bands(cls, sqft):
        area = pd.to_numeric(sqft, errors="coerce").to_numpy(dtype=float)
        bands = np.full(len(area), -1, dtype=int)
        valid = np.isfinite(area) & (area > 0)
        bands[valid] = np.searchsorted(cls.AREA_BOUNDARIES, area[valid], side="right")
        return bands

    def transform(self, homes):
        """Return numeric features aligned with homes' rows, without using answers."""
        if not hasattr(self, "history"):
            raise RuntimeError("Call ComparableSalesFeatures.fit(sales) first")

        result = pd.DataFrame(index=homes.index)
        n = len(homes)
        price_per_sqft = np.full(n, np.nan, dtype=float)
        count = np.zeros(n, dtype=int)
        level = np.zeros(n, dtype=int)

        dates = pd.to_datetime(homes["date_listed"], errors="coerce")
        months = dates.dt.to_period("M")
        sqft = square_feet(homes["house_area"]).to_numpy(dtype=float)
        bands = self._area_bands(homes["house_area"].pipe(square_feet))

        # Each month is a separate historical snapshot. A December listing,
        # for example, may use sales through November, never December sales.
        for month in months.dropna().unique():
            positions = np.flatnonzero((months == month).to_numpy())
            cutoff = month.to_timestamp()
            start = cutoff - pd.DateOffset(months=12)
            past = self.history.loc[
                (self.history["date_sold"] >= start)
                & (self.history["date_sold"] < cutoff)
            ]
            if past.empty:
                continue

            targets = homes.iloc[positions][["community", "house_type_name"]].copy()
            targets["area_band"] = bands[positions]
            for col in ("community", "house_type_name"):
                targets[col] = targets[col].astype("string").fillna("<missing>")
            targets = targets.reset_index(drop=True)

            selected = np.full(len(positions), np.nan, dtype=float)
            selected_count = np.zeros(len(positions), dtype=int)
            selected_level = np.zeros(len(positions), dtype=int)

            # Use specific comparables when there are at least 5. Otherwise
            # fall back to broader groups, so sparse locations still get data.
            groups = [
                (["community", "house_type_name", "area_band"], 5),
                (["community", "house_type_name"], 4),
                (["house_type_name", "area_band"], 3),
                (["house_type_name"], 2),
            ]
            for cols, specificity in groups:
                summary = (past.groupby(cols, dropna=False)["price_per_sqft"]
                           .agg(median="median", n="size").reset_index())
                matched = targets[cols].merge(summary, on=cols, how="left", sort=False)
                good = ((selected_level == 0)
                        & (matched["n"].fillna(0).to_numpy() >= self.MIN_COMPS))
                selected[good] = matched.loc[good, "median"].to_numpy(dtype=float)
                selected_count[good] = matched.loc[good, "n"].to_numpy(dtype=int)
                selected_level[good] = specificity

            # Global historical fallback, also strictly before the target month.
            if len(past) >= self.MIN_COMPS:
                fallback = selected_level == 0
                selected[fallback] = past["price_per_sqft"].median()
                selected_count[fallback] = len(past)
                selected_level[fallback] = 1

            price_per_sqft[positions] = selected
            count[positions] = selected_count
            level[positions] = selected_level

        result["past_ppsf_12m"] = price_per_sqft
        result["past_count_12m"] = count
        result["past_level_12m"] = level
        result["past_value_12m"] = price_per_sqft * sqft
        return result
