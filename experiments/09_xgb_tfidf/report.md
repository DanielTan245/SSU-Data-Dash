# Validation report

Walk-forward validation, 2025-12 to 2026-04: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 7,778 | 10.94% (10.72 to 11.16) | 8.52% (8.30 to 8.80) | -0.04% |
| list_aware | 7,778 | 5.50% (5.39 to 5.60) | 4.35% (4.23 to 4.45) | +0.56% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)
