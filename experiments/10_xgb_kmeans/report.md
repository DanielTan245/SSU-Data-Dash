# Validation report

Walk-forward validation, 2025-12 to 2026-04: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 7,778 | 11.86% (11.60 to 12.12) | 9.19% (8.96 to 9.41) | -0.28% |
| list_aware | 7,778 | 5.55% (5.44 to 5.66) | 4.33% (4.22 to 4.44) | +0.43% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](../figures/error_by_price_range.png)
