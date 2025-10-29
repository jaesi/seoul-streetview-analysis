"""
Basic Usage Example for Seoul Street View Analysis

This example demonstrates how to use the main modules of the project.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def example_modeling():
    """
    Example: Train ML models and make predictions.
    """
    print("=" * 80)
    print("EXAMPLE 1: Machine Learning Modeling")
    print("=" * 80)

    from src.modeling import UVIPredictor

    # Create predictor
    predictor = UVIPredictor()

    # Prepare data
    print("\n1. Loading data...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        segmentation_csv="class_percentages.csv",
        uvi_excel="Urban_vitality_index.xlsx"
    )

    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Features: {list(X_train.columns)[:5]}...")  # Show first 5

    # Train models
    print("\n2. Training models... (this may take a few minutes)")
    predictor.train_models(X_train, y_train)

    # Evaluate
    print("\n3. Evaluating models...")
    predictor.evaluate_models(X_train, X_test, y_train, y_test)

    # Print results
    print("\n4. Results:")
    predictor.print_results()

    # Make predictions
    print("\n5. Making predictions with best model...")
    predictions = predictor.predict(X_test, model_name='Gradient Boosting')
    print(f"   Sample predictions: {predictions[:5]}")

    print("\n✅ Example completed successfully!")


def example_segmentation():
    """
    Example: Segment a single image.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Image Segmentation")
    print("=" * 80)

    from src.segmenter import StreetViewSegmenter
    import os

    # Create segmenter
    segmenter = StreetViewSegmenter()

    # Find a sample image
    sample_folders = ["hongdae", "syarosu", "ssook", "test_images"]
    sample_image = None

    for folder in sample_folders:
        folder_path = os.path.join(".", folder)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if images:
                sample_image = os.path.join(folder_path, images[0])
                break

    if not sample_image:
        print("\n⚠️  No sample images found. Please run image collection first.")
        return

    print(f"\n1. Analyzing image: {sample_image}")

    # Segment image
    percentages = segmenter.segment_image(sample_image)

    # Print results
    print("\n2. Segmentation results:")
    print("   " + "-" * 40)
    for feature, percent in percentages.items():
        if percent > 0.1:  # Only show non-zero features
            bar = '█' * int(percent / 2)  # Visual bar
            print(f"   {feature:20s}: {percent:6.2f}% {bar}")
    print("   " + "-" * 40)

    print("\n✅ Example completed successfully!")


def example_custom_location():
    """
    Example: Configure a custom location for analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Location Configuration")
    print("=" * 80)

    print("\nTo analyze a new location, add it to config/settings.py:")
    print("""
    LOCATIONS["gangnam"] = {
        "name": "Gangnam Station (강남역)",
        "start_coords": (37.498, 127.027),
        "end_coords": (37.502, 127.030),
        "num_points": 30,
        "description": "Busy business district"
    }
    """)

    print("\nThen use it in your code:")
    print("""
    from config.settings import LOCATIONS
    from src.image_getter import StreetViewImageGetter

    location = LOCATIONS["gangnam"]
    getter = StreetViewImageGetter(api_key="YOUR_API_KEY")
    getter.fetch_images_along_path(
        start_lat=location["start_coords"][0],
        start_lon=location["start_coords"][1],
        end_lat=location["end_coords"][0],
        end_lon=location["end_coords"][1],
        num_points=location["num_points"],
        output_folder=f"data/raw/{location['name']}"
    )
    """)

    print("\n✅ Configuration example complete!")


def main():
    """Run all examples."""
    print("\n🚀 Seoul Street View Analysis - Basic Usage Examples")
    print("=" * 80)

    try:
        # Example 1: ML Modeling
        example_modeling()

        # Example 2: Image Segmentation
        example_segmentation()

        # Example 3: Custom Location
        example_custom_location()

        print("\n" + "=" * 80)
        print("🎉 All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease make sure:")
        print("  1. All dependencies are installed (pip install -r requirements.txt)")
        print("  2. Data files exist (class_percentages.csv, Urban_vitality_index.xlsx)")
        print("  3. You're running from the project root directory")


if __name__ == "__main__":
    main()
