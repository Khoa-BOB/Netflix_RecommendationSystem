import pandas as pd
import argparse
from collab_filtering.memory_based_cf import MemoryBasedCF
from collab_filtering.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train a collaboratve filtering recommendation system")

    parser.add_argument("--k", type=int, default=10, help="neighborhood size")
    parser.add_argument("--metrics", choices=["cosine", "pearson"], default="cosine", help="similarity metrics: Cosine or Pearson Correlation")
    parser.add_argument("--cf_type", choices=["user","item"], default="item", help="\'item\' for item-based, \'user\' for user-based")

    args = parser.parse_args()

    print(args.k,args.metrics,args.cf_type)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv("netflix/ratings-train.txt",
                          header=None,
                          names=["Movie-id", "User-id", "rating"])
    test_df = pd.read_csv("netflix/ratings-test.txt",
                         header=None,
                         names=["Movie-id", "User-id", "rating"])
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}\n")

    model = MemoryBasedCF(k=args.k, similarity_metric=args.metrics, cf_type=args.cf_type)

    model.fit(train_df)
    results = evaluate_model(model=model, test_df=test_df)

if __name__ == "__main__":
    main()
