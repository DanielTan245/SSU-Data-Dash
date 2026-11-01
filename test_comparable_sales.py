"""Run: uv run python -m unittest test_comparable_sales.py -v"""
import unittest
import numpy as np
import pandas as pd
from comparable_sales import ComparableSalesFeatures


def sale(ml, sold, listed, price, community="A", typ="Detached", area="1000"):
    return {"ml_num":ml, "date_sold":pd.Timestamp(sold),
            "date_listed":pd.Timestamp(listed), "sold_price":price,
            "community":community, "house_type_name":typ, "house_area":area}


class ComparableFeatureTests(unittest.TestCase):
    def setUp(self):
        self.sales = pd.DataFrame([
            *[sale(f"old_{i}", "2025-01-15", "2024-12-10", 100_000) for i in range(6)],
            # Same month as target listing: forbidden, even if sold earlier in the month.
            sale("feb_early", "2025-02-01", "2025-01-17", 9_000_000),
            # The target home itself is in training history: must never see its label.
            sale("target", "2025-02-20", "2025-02-10", 8_000_000),
            # Future home: never allowed when predicting February.
            sale("future", "2025-03-05", "2025-02-26", 7_000_000),
        ])

    def test_no_same_month_future_or_self_sale(self):
        transform = ComparableSalesFeatures().fit(self.sales)
        homes = self.sales.iloc[[7]].copy()  # target in history, with sold price present
        x = transform.transform(homes)
        self.assertAlmostEqual(x["past_ppsf_12m"].iloc[0], 100.0)
        self.assertEqual(x["past_count_12m"].iloc[0], 6)
        self.assertEqual(x["past_level_12m"].iloc[0], 5)
        self.assertAlmostEqual(x["past_value_12m"].iloc[0], 100_000)

    def test_no_future_sales_for_old_listing(self):
        transform = ComparableSalesFeatures().fit(self.sales)
        old = pd.DataFrame([{"date_listed":pd.Timestamp("2024-06-02"), "community":"A",
                             "house_type_name":"Detached", "house_area":"1000"}])
        x = transform.transform(old)
        self.assertTrue(np.isnan(x["past_ppsf_12m"].iloc[0]))
        self.assertEqual(x["past_count_12m"].iloc[0], 0)

    def test_fallback_to_property_type_if_sparse_community(self):
        transform = ComparableSalesFeatures().fit(self.sales)
        unknown = pd.DataFrame([{"date_listed":pd.Timestamp("2025-02-09"), "community":"B",
                                 "house_type_name":"Detached", "house_area":"1000"}])
        x = transform.transform(unknown)
        self.assertAlmostEqual(x["past_ppsf_12m"].iloc[0], 100.0)
        self.assertEqual(x["past_level_12m"].iloc[0], 3)

    def test_older_than_twelve_months_is_excluded(self):
        old = [sale(f"ancient_{i}", "2023-11-20", "2023-10-10", 100_000) for i in range(6)]
        recent = [sale(f"recent_{i}", "2025-01-10", "2024-12-05", 200_000) for i in range(6)]
        builder = ComparableSalesFeatures().fit(pd.DataFrame(old + recent))
        home = pd.DataFrame([{"date_listed":pd.Timestamp("2025-02-10"), "community":"A",
                              "house_type_name":"Detached", "house_area":"1000"}])
        x = builder.transform(home)
        self.assertAlmostEqual(x["past_ppsf_12m"].iloc[0], 200.0)
        self.assertEqual(x["past_count_12m"].iloc[0], 6)

    def test_no_future_training_leakage(self):
        old = sale("first", "2024-02-03", "2024-01-02", 1_000_000)
        later = [sale(f"later_{i}", "2024-08-01", "2024-07-01", 3_000_000)
                 for i in range(6)]
        builder = ComparableSalesFeatures().fit(pd.DataFrame([old, *later]))
        x = builder.transform(pd.DataFrame([old]))
        self.assertTrue(np.isnan(x["past_ppsf_12m"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
