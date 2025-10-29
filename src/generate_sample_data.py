"""
Generate Sample Data

This script generates sample data for testing the pipeline when Google API key
or actual street view images are not available.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import random


def generate_sample_image(
    width: int = 256,
    height: int = 256,
    save_path: str = None
) -> Image:
    """
    Generate a sample street view-like image.

    Args:
        width: Image width (default: 256)
        height: Image height (default: 256)
        save_path: Path to save the image (optional)

    Returns:
        PIL Image object
    """
    # Create a new image
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Sky (top 1/3)
    sky_color = (135 + random.randint(-20, 20),
                 206 + random.randint(-20, 20),
                 235 + random.randint(-20, 20))
    draw.rectangle([0, 0, width, height // 3], fill=sky_color)

    # Buildings (middle section)
    num_buildings = random.randint(2, 5)
    building_y_start = height // 4

    for i in range(num_buildings):
        building_width = width // num_buildings
        x1 = i * building_width
        x2 = (i + 1) * building_width
        building_height = random.randint(height // 3, height * 2 // 3)
        y1 = building_y_start
        y2 = y1 + building_height

        building_color = (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200)
        )
        draw.rectangle([x1, y1, x2, y2], fill=building_color)

        # Windows
        window_rows = random.randint(3, 8)
        window_cols = random.randint(2, 4)
        window_width = building_width // (window_cols + 1)
        window_height = building_height // (window_rows + 1)

        for row in range(window_rows):
            for col in range(window_cols):
                wx1 = x1 + (col + 1) * window_width - window_width // 2
                wy1 = y1 + (row + 1) * window_height - window_height // 2
                wx2 = wx1 + window_width // 2
                wy2 = wy1 + window_height // 2
                window_color = (255, 255, 200) if random.random() > 0.3 else (50, 50, 50)
                draw.rectangle([wx1, wy1, wx2, wy2], fill=window_color)

    # Road (bottom section)
    road_y_start = height * 3 // 4
    road_color = (70 + random.randint(-10, 10),
                  70 + random.randint(-10, 10),
                  70 + random.randint(-10, 10))
    draw.rectangle([0, road_y_start, width, height], fill=road_color)

    # Road markings
    for i in range(0, width, 40):
        draw.rectangle([i + 10, height - 30, i + 25, height - 25],
                       fill=(255, 255, 255))

    # Sidewalk
    sidewalk_height = 20
    sidewalk_color = (200, 200, 200)
    draw.rectangle([0, road_y_start - sidewalk_height, width, road_y_start],
                   fill=sidewalk_color)

    # Add some green (trees/vegetation)
    num_trees = random.randint(1, 3)
    for _ in range(num_trees):
        tree_x = random.randint(0, width)
        tree_y = random.randint(height // 2, road_y_start)
        tree_radius = random.randint(10, 20)
        tree_color = (34 + random.randint(-10, 10),
                      139 + random.randint(-30, 30),
                      34 + random.randint(-10, 10))
        draw.ellipse([tree_x - tree_radius, tree_y - tree_radius,
                      tree_x + tree_radius, tree_y + tree_radius],
                     fill=tree_color)

    if save_path:
        img.save(save_path)

    return img


def generate_sample_images_for_location(
    location_name: str,
    num_points: int,
    output_folder: str,
    headings: list = [0, 90, 180, 270]
) -> None:
    """
    Generate sample images for a location.

    Args:
        location_name: Name of the location
        num_points: Number of points along the path
        output_folder: Folder to save images
        headings: List of camera headings
    """
    os.makedirs(output_folder, exist_ok=True)

    total_images = num_points * len(headings)
    print(f"Generating {total_images} sample images for {location_name}...")

    for i in range(num_points):
        for heading in headings:
            filename = f"{location_name}_image_{i}_heading_{heading}.jpg"
            filepath = os.path.join(output_folder, filename)
            generate_sample_image(save_path=filepath)

    print(f"Generated {total_images} images in {output_folder}")


def generate_sample_segmentation_data(
    image_folder: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Generate sample segmentation data based on image filenames.

    Args:
        image_folder: Folder containing images
        output_csv: Path to save CSV file

    Returns:
        DataFrame with sample segmentation data
    """
    import glob

    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    if not image_files:
        print(f"No images found in {image_folder}")
        return pd.DataFrame()

    data = []

    for img_path in image_files:
        filename = os.path.basename(img_path)

        # Generate realistic-looking percentages that sum to ~100%
        percentages = {
            'filename': filename,
            'unlabelled': random.uniform(5, 20),
            'ground': random.uniform(5, 25),
            'building': random.uniform(15, 50),
            'road': random.uniform(0.5, 10),
            'green': random.uniform(0.5, 10),
            'sky': random.uniform(5, 25),
            'pedestrian': random.uniform(1, 30),
            'building2': random.uniform(5, 35)
        }

        # Normalize to sum to 100
        total = sum(v for k, v in percentages.items() if k != 'filename')
        for key in percentages:
            if key != 'filename':
                percentages[key] = (percentages[key] / total) * 100

        data.append(percentages)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated segmentation data: {output_csv}")

    return df


def generate_sample_uvi_data(
    num_samples: int,
    output_excel: str
) -> pd.DataFrame:
    """
    Generate sample UVI (Urban Vitality Index) data.

    Args:
        num_samples: Number of samples
        output_excel: Path to save Excel file

    Returns:
        DataFrame with UVI data
    """
    # Generate UVI values between 3 and 9 (typical range)
    uvi_values = [random.randint(3, 9) for _ in range(num_samples)]

    df = pd.DataFrame(uvi_values)
    df.to_excel(output_excel, index=False, header=False)
    print(f"Generated UVI data: {output_excel}")

    return df


def main():
    """Generate all sample data."""
    print("=" * 60)
    print("GENERATING SAMPLE DATA FOR SEOUL STREETVIEW ANALYSIS")
    print("=" * 60)

    # Define locations
    locations = {
        "hongdae": {"points": 40, "output": "data/raw/hongdae"},
        "syarosu": {"points": 20, "output": "data/raw/syarosu"},
        "ssook": {"points": 20, "output": "data/raw/ssook"}
    }

    # Generate sample images
    print("\n1. Generating sample street view images...")
    for location_name, config in locations.items():
        generate_sample_images_for_location(
            location_name=location_name,
            num_points=config["points"],
            output_folder=config["output"]
        )

    # Combine all locations for segmentation data
    print("\n2. Generating sample segmentation data...")

    all_data = []
    for location_name, config in locations.items():
        df = generate_sample_segmentation_data(
            image_folder=config["output"],
            output_csv=f"data/processed/{location_name}_segmentation.csv"
        )
        all_data.append(df)

    # Combine all segmentation data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("class_percentages.csv", index=False)
    print(f"Generated combined segmentation data: class_percentages.csv")

    # Generate sample UVI data
    print("\n3. Generating sample UVI data...")
    num_samples = len(combined_df)
    generate_sample_uvi_data(num_samples, "Urban_vitality_index.xlsx")

    # Generate a smaller test set
    print("\n4. Generating test data...")
    test_location = "hongdae"
    test_images_folder = "test_images"
    os.makedirs(test_images_folder, exist_ok=True)

    # Generate 2 test images (8 total with 4 headings)
    generate_sample_images_for_location(
        location_name="hongdae",
        num_points=2,
        output_folder=test_images_folder,
        headings=[0, 90, 180, 270]
    )

    generate_sample_segmentation_data(
        image_folder=test_images_folder,
        output_csv="class_percentages_test.csv"
    )

    print("\n" + "=" * 60)
    print("SAMPLE DATA GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - Sample images in data/raw/")
    print("  - class_percentages.csv")
    print("  - Urban_vitality_index.xlsx")
    print("  - class_percentages_test.csv")
    print("  - Test images in test_images/")


if __name__ == "__main__":
    main()
