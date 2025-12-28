# test_predict.py

import argparse
import os
import pandas as pd

from src.pipeline import run_inference_on_folder


def main(test_dir: str, output_path: str):
    if not os.path.isdir(test_dir):
        raise ValueError(f"Test directory does not exist: {test_dir}")

    # Run inference
    results = run_inference_on_folder(test_dir)

    # Create DataFrame
    df = pd.DataFrame(results, columns=["image_name", "prediction"])

    # Save results
    df.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to: {output_path}")
    print(f"Total images processed: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run car orientation prediction on a test folder"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path to folder containing test images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions CSV"
    )

    args = parser.parse_args()

    main(args.test_dir, args.output)
