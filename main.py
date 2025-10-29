"""
Main Pipeline Script

This script runs the complete Seoul Street View Analysis pipeline.
"""

import argparse
import sys
from pathlib import Path


def run_full_pipeline():
    """Run the complete analysis pipeline."""
    print("=" * 80)
    print("SEOUL STREET VIEW ANALYSIS - FULL PIPELINE")
    print("=" * 80)

    # Step 1: Check if data exists
    print("\n[Step 1/3] Checking data...")
    if not Path("class_percentages.csv").exists() or not Path("Urban_vitality_index.xlsx").exists():
        print("Data files not found. Please run one of the following:")
        print("  - python src/image_getter.py (if you have Google API key)")
        print("  - python src/generate_sample_data.py (to generate sample data)")
        return

    # Step 2: Run modeling
    print("\n[Step 2/3] Training and evaluating ML models...")
    from src.modeling import main as run_modeling
    run_modeling()

    # Step 3: Complete
    print("\n[Step 3/3] Pipeline complete!")
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("- Model performance: See output above")
    print("- Feature importance plot: data/processed/feature_importance.png")
    print("\nYou can now:")
    print("  1. Review the model results")
    print("  2. Check feature importance visualization")
    print("  3. Use trained models for predictions")


def run_image_collection():
    """Run image collection using Google Maps API."""
    from src.image_getter import main as run_getter
    print("Starting image collection...")
    run_getter()


def run_segmentation():
    """Run image segmentation."""
    from src.segmenter import main as run_segmenter
    print("Starting image segmentation...")
    run_segmenter()


def run_modeling():
    """Run ML modeling."""
    from src.modeling import main as run_model
    print("Starting ML modeling...")
    run_model()


def generate_sample_data():
    """Generate sample data for testing."""
    from src.generate_sample_data import main as gen_data
    print("Generating sample data...")
    gen_data()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Seoul Street View Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full              # Run complete pipeline
  python main.py --collect-images    # Fetch images from Google Maps
  python main.py --segment           # Run segmentation
  python main.py --model             # Train ML models
  python main.py --generate-data     # Generate sample data
        """
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Run the complete pipeline'
    )
    parser.add_argument(
        '--collect-images',
        action='store_true',
        help='Collect images using Google Maps API'
    )
    parser.add_argument(
        '--segment',
        action='store_true',
        help='Run image segmentation'
    )
    parser.add_argument(
        '--model',
        action='store_true',
        help='Train and evaluate ML models'
    )
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate sample data for testing'
    )

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Execute based on arguments
    if args.full:
        run_full_pipeline()
    elif args.collect_images:
        run_image_collection()
    elif args.segment:
        run_segmentation()
    elif args.model:
        run_modeling()
    elif args.generate_data:
        generate_sample_data()


if __name__ == "__main__":
    main()
