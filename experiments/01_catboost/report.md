# Validation report

Walk-forward validation, 2025-12 to 2026-04: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 7,778 | 11.79% (11.57 to 12.02) | 9.48% (9.23 to 9.73) | -0.86% |
| list_aware | 7,778 | 5.41% (5.31 to 5.52) | 4.24% (4.13 to 4.36) | +0.37% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)
