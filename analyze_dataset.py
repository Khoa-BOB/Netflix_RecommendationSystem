import pandas as pd
import numpy as np

# Load training data
training_file = "netflix/ratings-train.txt"
ratings = pd.read_csv(training_file, header=None, names=["Movie-id", "User-id", "rating"])

print("=== DATASET OVERVIEW ===")
print(f"Total ratings: {len(ratings)}")
print(f"Unique movies: {ratings['Movie-id'].nunique()}")
print(f"Unique users: {ratings['User-id'].nunique()}")

# Calculate sparsity
n_users = ratings['User-id'].nunique()
n_movies = ratings['Movie-id'].nunique()
n_possible = n_users * n_movies
sparsity = 100 * (1 - len(ratings) / n_possible)

print(f"\n=== SPARSITY ANALYSIS ===")
print(f"Matrix dimensions: {n_users} users × {n_movies} movies")
print(f"Possible ratings: {n_possible}")
print(f"Actual ratings: {len(ratings)}")
print(f"Sparsity: {sparsity:.2f}%")

# Ratings per movie
print(f"\n=== RATINGS PER MOVIE ===")
movie_counts = ratings['Movie-id'].value_counts()
print(f"Mean: {movie_counts.mean():.1f}")
print(f"Median: {movie_counts.median():.1f}")
print(f"Min: {movie_counts.min()}")
print(f"Max: {movie_counts.max()}")
print(f"Std: {movie_counts.std():.1f}")

# Ratings per user
print(f"\n=== RATINGS PER USER ===")
user_counts = ratings['User-id'].value_counts()
print(f"Mean: {user_counts.mean():.1f}")
print(f"Median: {user_counts.median():.1f}")
print(f"Min: {user_counts.min()}")
print(f"Max: {user_counts.max()}")
print(f"Std: {user_counts.std():.1f}")

# Rating distribution
print(f"\n=== RATING VALUES DISTRIBUTION ===")
print(ratings['rating'].value_counts().sort_index())

# Check for cold start issues
print(f"\n=== POTENTIAL ISSUES ===")
movies_few_ratings = (movie_counts < 10).sum()
users_few_ratings = (user_counts < 3).sum()
print(f"Movies with < 10 ratings: {movies_few_ratings} ({100*movies_few_ratings/n_movies:.1f}%)")
print(f"Users with < 3 ratings: {users_few_ratings} ({100*users_few_ratings/n_users:.1f}%)")

# Check overlap for collaborative filtering
print(f"\n=== COLLABORATIVE FILTERING SUITABILITY ===")
users_with_multiple = (user_counts >= 2).sum()
movies_with_multiple = (movie_counts >= 2).sum()
print(f"Users with 2+ ratings: {users_with_multiple} ({100*users_with_multiple/n_users:.1f}%)")
print(f"Movies with 2+ ratings: {movies_with_multiple} ({100*movies_with_multiple/n_movies:.1f}%)")
