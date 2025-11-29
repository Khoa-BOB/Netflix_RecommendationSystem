"""
Run experiments with different neighborhood sizes (k) and report MSE
"""
import pandas as pd
import json
from datetime import datetime
from collab_filtering.memory_based_cf import MemoryBasedCF
from collab_filtering.evaluation import evaluate_model

def run_experiments():
    # Different k values to test
    k_values = [3, 6, 9, 12, 15, 20]

    # Test both similarity metrics
    similarity_metrics = ['cosine', 'pearson']

    # Test both CF types
    cf_types = ['item', 'user']

    # Load data once
    print("Loading data...")
    train_df = pd.read_csv("netflix/ratings-train.txt",
                          header=None,
                          names=["Movie-id", "User-id", "rating"])
    test_df = pd.read_csv("netflix/ratings-test.txt",
                         header=None,
                         names=["Movie-id", "User-id", "rating"])

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}\n")

    # Store results
    results = []

    # Run experiment for each combination
    for cf_type in cf_types:
        for similarity in similarity_metrics:
            print(f"\n{'='*70}")
            print(f"Testing {cf_type.upper()}-based CF with {similarity.upper()} similarity")
            print(f"{'='*70}")

            for k in k_values:
                print(f"\n{'-'*70}")
                print(f"Running experiment: CF={cf_type}, Similarity={similarity}, k={k}")
                print(f"{'-'*70}")

                # Create and train model
                model = MemoryBasedCF(k=k, similarity_metric=similarity, cf_type=cf_type)
                model.fit(train_df)

                # Evaluate
                eval_results = evaluate_model(model=model, test_df=test_df)

                # Store results
                result = {
                    'cf_type': cf_type,
                    'similarity': similarity,
                    'k': k,
                    'mse': eval_results['mse'],
                    'rmse': eval_results['rmse'],
                    'mae': eval_results['mae']
                }
                results.append(result)

                print(f"\nResults for {cf_type}-based, {similarity}, k={k}:")
                print(f"  MSE:  {eval_results['mse']:.4f}")
                print(f"  RMSE: {eval_results['rmse']:.4f}")
                print(f"  MAE:  {eval_results['mae']:.4f}")

    # Print summary
    print(f"\n\n{'='*90}")
    print("SUMMARY: MSE for Different Configurations")
    print(f"{'='*90}")

    # Group results by CF type and similarity
    for cf_type in cf_types:
        for similarity in similarity_metrics:
            print(f"\n{cf_type.upper()}-based CF with {similarity.upper()} similarity:")
            print(f"{'-'*90}")
            print(f"{'k':<10} {'MSE':<20} {'RMSE':<20} {'MAE':<20}")
            print(f"{'-'*90}")

            # Filter results for this combination
            filtered_results = [r for r in results if r['cf_type'] == cf_type and r['similarity'] == similarity]

            for result in filtered_results:
                print(f"{result['k']:<10} {result['mse']:<20.4f} {result['rmse']:<20.4f} {result['mae']:<20.4f}")

    # Find best configuration
    best_result = min(results, key=lambda x: x['mse'])
    print(f"\n\n{'='*90}")
    print("BEST CONFIGURATION (Lowest MSE)")
    print(f"{'='*90}")
    print(f"CF Type:     {best_result['cf_type'].upper()}-based")
    print(f"Similarity:  {best_result['similarity'].upper()}")
    print(f"k:           {best_result['k']}")
    print(f"MSE:         {best_result['mse']:.4f}")
    print(f"RMSE:        {best_result['rmse']:.4f}")
    print(f"MAE:         {best_result['mae']:.4f}")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"k_experiments_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*90}")
    print(f"Results saved to: {output_file}")

    # Also save as CSV for easier viewing
    csv_file = f"k_experiments_results_{timestamp}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_file, index=False)
    print(f"Results also saved to: {csv_file}")
    print(f"{'='*90}")

    return results

if __name__ == "__main__":
    results = run_experiments()
