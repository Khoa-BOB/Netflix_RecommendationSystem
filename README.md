# Netflix Recommendation System - Collaborative Filtering

## Results Summary

### ITEM-based CF with COSINE similarity

| k  | MSE   | RMSE  | MAE   |
|----|-------|-------|-------|
| 3  | 1.537 | 1.240 | 0.933 |
| 6  | 1.529 | 1.236 | 0.934 |
| 9  | 1.534 | 1.238 | 0.937 |
| 12 | 1.531 | 1.237 | 0.936 |
| 15 | 1.530 | 1.237 | 0.935 |
| 20 | 1.530 | 1.237 | 0.934 |

**Best k for Item-Cosine: k=20 (MSE: 1.530)**

---

### ITEM-based CF with PEARSON similarity

| k  | MSE   | RMSE  | MAE   |
|----|-------|-------|-------|
| 3  | 1.560 | 1.249 | 0.940 |
| 6  | 1.552 | 1.246 | 0.934 |
| 9  | 1.556 | 1.248 | 0.937 |
| 12 | 1.555 | 1.247 | 0.938 |
| 15 | 1.554 | 1.246 | 0.938 |
| 20 | 1.552 | 1.246 | 0.937 |

**Best k for Item-Pearson: k=20 (MSE: 1.552)**

---

### USER-based CF with COSINE similarity

| k  | MSE   | RMSE  | MAE   |
|----|-------|-------|-------|
| 3  | 1.663 | 1.290 | 1.015 |
| 6  | 1.622 | 1.274 | 1.012 |
| 9  | 1.620 | 1.273 | 1.013 |
| 12 | 1.629 | 1.276 | 1.020 |
| 15 | 1.624 | 1.274 | 1.021 |
| 20 | 1.613 | 1.270 | 1.018 |

**Best k for User-Cosine: k=20 (MSE: 1.613)**

---

### USER-based CF with PEARSON similarity

| k  | MSE   | RMSE  | MAE   |
|----|-------|-------|-------|
| 3  | 1.635 | 1.279 | 1.002 |
| 6  | 1.624 | 1.274 | 0.999 |
| 9  | 1.626 | 1.275 | 0.999 |
| 12 | 1.626 | 1.275 | 0.999 |
| 15 | 1.626 | 1.275 | 0.999 |
| 20 | 1.626 | 1.275 | 0.999 |

**Best k for User-Pearson: k=6 (MSE: 1.624)**

---

## Overall Best Configuration

**Configuration:** ITEM-based CF with COSINE similarity
**k:** 20
**MSE:** 1.530
**RMSE:** 1.237
**MAE:** 0.934

---

## Key Observations

1. **Item-based CF outperforms User-based CF** across all configurations
2. **Cosine similarity performs slightly better than Pearson** for both CF types
3. **MSE generally improves with larger k values**, with diminishing returns after k=6
4. **Item-Cosine with k=20** achieved the lowest MSE of 1.530
5. User-based methods show higher error metrics (MSE ~1.6) compared to item-based (MSE ~1.5)

---

*Results generated from: k_experiments_results_20251125_073304.json*
