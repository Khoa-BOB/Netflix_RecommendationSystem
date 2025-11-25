"""
Memory-Based Collaborative Filtering Implementation
Implements both User-Based and Item-Based CF
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cosine_similarity_manual(vec1, vec2):
    """
    Calculate cosine similarity between two vectors, ignoring NaN values

    similarity = (A · B) / (||A|| * ||B||)
    """
    # Find common rated items (non-NaN in both vectors)
    mask = ~(np.isnan(vec1) | np.isnan(vec2))

    if mask.sum() == 0:
        return 0.0

    # Boolean masking to filter non NaN
    v1 = vec1[mask]
    v2 = vec2[mask]

    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def pearson_correlation_manual(vec1, vec2):
    """
    Calculate Pearson correlation between two vectors, ignoring NaN values

    correlation = covariance(A, B) / (std(A) * std(B))
    """
    # Find common rated items
    mask = ~(np.isnan(vec1) | np.isnan(vec2))

    if mask.sum() < 2:  # Need at least 2 common ratings
        return 0.0

    v1 = vec1[mask]
    v2 = vec2[mask]

    # Calculate means
    mean1 = v1.mean()
    mean2 = v2.mean()

    # Calculate Pearson correlation
    numerator = ((v1 - mean1) * (v2 - mean2)).sum()
    denominator = np.sqrt(((v1 - mean1)**2).sum() * ((v2 - mean2)**2).sum())

    if denominator == 0:
        return 0.0

    return numerator / denominator


class MemoryBasedCF:
    def __init__(self, k=10, similarity_metric='cosine', cf_type='item'):
        """
        Memory-Based Collaborative Filtering

        k: number of neighbors to consider
        similarity_metric: 'cosine' or 'pearson'
        cf_type: 'item' for item-based, 'user' for user-based
        """
        self.k = k
        self.similarity_metric = similarity_metric
        self.cf_type = cf_type
        self.user_item_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None

    def fit(self, ratings_df):
        """
        Build the user-item matrix from training data

        ratings_df: DataFrame with columns ['Movie-id', 'User-id', 'rating']
        """
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='User-id',
            columns='Movie-id',
            values='rating'
        )

        print(f"User-Item Matrix shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {(self.user_item_matrix.isna().sum().sum() / self.user_item_matrix.size) * 100:.2f}%")

        # Calculate mean ratings
        self.user_means = self.user_item_matrix.mean(axis=1)
        self.item_means = self.user_item_matrix.mean(axis=0)
        self.global_mean = ratings_df['rating'].mean()

        print(f"CF Type: {self.cf_type.upper()}-based")
        print(f"Similarity Metric: {self.similarity_metric}")
        print(f"Number of neighbors (k): {self.k}")

    def calculate_similarity(self, vec1, vec2):
        """Calculate similarity between two vectors"""
        if self.similarity_metric == 'cosine':
            return cosine_similarity_manual(vec1, vec2)
        elif self.similarity_metric == 'pearson':
            return pearson_correlation_manual(vec1, vec2)
        else:
            raise ValueError("Similarity metric must be 'cosine' or 'pearson'")

    def predict_item_based(self, user_id, movie_id):
        """
        Item-based prediction: Find similar movies and aggregate ratings
        """
        # Check if user/movie exists
        if user_id not in self.user_item_matrix.index:
            return self.global_mean

        if movie_id not in self.user_item_matrix.columns:
            return self.user_means[user_id] if not np.isnan(self.user_means[user_id]) else self.global_mean

        # Get the target movie's ratings across all users
        target_movie = self.user_item_matrix[movie_id].values

        # Get movies rated by this user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings.notna()]

        if len(rated_movies) == 0:
            return self.item_means[movie_id]

        # Calculate similarity between target movie and all rated movies
        similarities = []
        for rated_movie_id in rated_movies.index:
            if rated_movie_id == movie_id:
                continue

            rated_movie = self.user_item_matrix[rated_movie_id].values
            sim = self.calculate_similarity(target_movie, rated_movie)

            if sim > 0:  # Only consider positive similarities
                similarities.append({
                    'movie_id': rated_movie_id,
                    'similarity': sim,
                    'rating': rated_movies[rated_movie_id]
                })

        if len(similarities) == 0:
            return self.user_means[user_id]

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_k = similarities[:self.k]

        # Weighted average
        numerator = sum(item['similarity'] * item['rating'] for item in top_k)
        denominator = sum(item['similarity'] for item in top_k)

        if denominator == 0:
            return self.user_means[user_id]

        prediction = numerator / denominator
        return np.clip(prediction, 1.0, 5.0)

    def predict_user_based(self, user_id, movie_id):
        """
        User-based prediction: Find similar users and aggregate ratings
        """
        # Check if user/movie exists
        if user_id not in self.user_item_matrix.index:
            return self.global_mean

        if movie_id not in self.user_item_matrix.columns:
            return self.item_means[movie_id] if not np.isnan(self.item_means[movie_id]) else self.global_mean

        # Get the target user's ratings
        target_user = self.user_item_matrix.loc[user_id].values

        # Get users who rated this movie
        movie_ratings = self.user_item_matrix[movie_id]
        users_who_rated = movie_ratings[movie_ratings.notna()]

        if len(users_who_rated) == 0:
            return self.user_means[user_id]

        # Calculate similarity between target user and all users who rated the movie
        similarities = []
        for other_user_id in users_who_rated.index:
            if other_user_id == user_id:
                continue

            other_user = self.user_item_matrix.loc[other_user_id].values
            sim = self.calculate_similarity(target_user, other_user)

            if sim > 0:
                similarities.append({
                    'user_id': other_user_id,
                    'similarity': sim,
                    'rating': users_who_rated[other_user_id]
                })

        if len(similarities) == 0:
            return self.item_means[movie_id]

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_k = similarities[:self.k]

        # Weighted average
        numerator = sum(item['similarity'] * item['rating'] for item in top_k)
        denominator = sum(item['similarity'] for item in top_k)

        if denominator == 0:
            return self.item_means[movie_id]

        prediction = numerator / denominator
        return np.clip(prediction, 1.0, 5.0)

    def predict(self, user_id, movie_id):
        """Predict rating based on CF type"""
        if self.cf_type == 'item':
            return self.predict_item_based(user_id, movie_id)
        else:
            return self.predict_user_based(user_id, movie_id)

    def predict_batch(self, test_df):
        """Predict ratings for multiple user-movie pairs"""
        predictions = []
        total = len(test_df)

        for idx, row in test_df.iterrows():
            if (idx + 1) % 50 == 0:
                print(f"Predicting {idx + 1}/{total}...")

            pred = self.predict(row['User-id'], row['Movie-id'])
            predictions.append(pred)

        return np.array(predictions)

    def recommend_for_user(self, user_id, n=5):
        """Recommend top n movies for a user"""
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found")
            return None

        # Get unrated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index

        # Predict ratings
        predictions = []
        for movie_id in unrated_movies:
            pred = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred))

        # Sort and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]

        return pd.DataFrame(top_n, columns=['Movie-id', 'Predicted Rating'])
