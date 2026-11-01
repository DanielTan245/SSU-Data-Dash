# Validation report

Walk-forward validation, 2025-12 to 2026-04: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 7,778 | 11.09% (10.87 to 11.31) | 8.82% (8.65 to 9.02) | -0.22% |
| list_aware | 7,778 | 5.50% (5.39 to 5.60) | 4.33% (4.22 to 4.43) | +0.52% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)
