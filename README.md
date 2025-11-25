# Netflix Recommendation System - Collaborative Filtering

## How to run

### Clone the repository
```bash
git clone https://github.com/Khoa-BOB/Netflix_RecommendationSystem.git
cd Netflix_RecommendationSystem
```

### Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the program

#### Basic usage (with default parameters)
```bash
python main.py
```

#### Run with custom parameters
```bash
python main.py --k <neighborhood_size> --metrics <similarity_metric> --cf_type <cf_type>
```

**Available parameters:**
- `--k`: Neighborhood size (default: 10)
  - Example values: 3, 6, 9, 12, 15, 20
- `--metrics`: Similarity metric (default: cosine)
  - Options: `cosine` or `pearson`
- `--cf_type`: Collaborative filtering type (default: item)
  - Options: `item` or `user`

**Examples:**
```bash
# Item-based CF with cosine similarity and k=20
python main.py --k 20 --metrics cosine --cf_type item

# User-based CF with pearson correlation and k=10
python main.py --k 10 --metrics pearson --cf_type user

# Item-based CF with pearson and k=15
python main.py --k 15 --metrics pearson --cf_type item
```

#### Run comprehensive experiments (all combinations)
To test multiple k values across all CF types and similarity metrics:
```bash
# For local testing
python run_k_experiments.py

# For HPC cluster submission (SLURM)
sbatch submit_k_experiments.sbatch
```

This will test 24 configurations and generate detailed results.

---

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

## Reference
[Empirical Analysis of Predictive Algorithms for Collaborative
Filtering](https://arxiv.org/ftp/arxiv/papers/1301/1301.7363.pdf)
