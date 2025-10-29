"""
Segmenter Module

This module provides functionality for image segmentation using ResNet50-based models.
"""

import os
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path


class StreetViewSegmenter:
    """Class for segmenting street view images using FCN-ResNet50."""

    def __init__(self):
        """Initialize the StreetViewSegmenter with pre-trained model."""
        self.weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def segment_image(
        self,
        image_path: str,
        threshold: float = 0.6
    ) -> Dict[str, float]:
        """
        Segment an image and return class percentages.

        Args:
            image_path: Path to the input image
            threshold: Confidence threshold for segmentation (default: 0.6)

        Returns:
            Dictionary mapping class names to their percentage in the image
        """
        # Load image
        img = read_image(image_path)
        img = img[:3, :, :]  # Convert RGBA to RGB if needed

        # Preprocess
        batch = self.preprocess(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            prediction = self.model(batch)["out"]
            normalized_masks = prediction.softmax(dim=1)

        # Calculate percentages for each class
        height, width = normalized_masks.shape[2:]
        total_pixels = height * width
        percentages = {}

        for idx, category in enumerate(self.categories):
            mask = normalized_masks[0, idx].numpy()
            mask = mask > threshold
            pixel_count = np.sum(mask)
            percentage = (pixel_count / total_pixels) * 100
            percentages[category] = percentage

        return percentages

    def segment_and_visualize(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        threshold: float = 0.6
    ) -> None:
        """
        Segment an image and save visualization.

        Args:
            image_path: Path to the input image
            save_path: Path to save the visualization (optional)
            threshold: Confidence threshold for segmentation (default: 0.6)
        """
        # Load image
        img = read_image(image_path)
        img = img[:3, :, :]

        # Preprocess
        batch = self.preprocess(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            prediction = self.model(batch)["out"]
            normalized_masks = prediction.softmax(dim=1)

        # Create overlay
        original_image = to_pil_image(batch[0])
        overlay = np.zeros((original_image.height, original_image.width, 4))

        # Apply masks with random colors
        for cls_idx in range(len(self.categories)):
            mask = normalized_masks[0, cls_idx].detach().numpy()
            mask = mask > threshold
            if np.any(mask):
                color = np.random.rand(3)
                overlay[mask] = np.append(color, 0.5)

        # Visualize
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay, interpolation='none')
        plt.title("Segmentation")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def process_folder(
        self,
        input_folder: str,
        output_csv: str,
        file_extension: str = ".jpg"
    ) -> pd.DataFrame:
        """
        Process all images in a folder and save results to CSV.

        Args:
            input_folder: Folder containing images
            output_csv: Path to save the CSV file
            file_extension: Extension of image files (default: ".jpg")

        Returns:
            DataFrame containing segmentation results
        """
        results = []
        image_files = sorted(Path(input_folder).glob(f"*{file_extension}"))

        print(f"Processing {len(image_files)} images from {input_folder}...")

        for image_path in image_files:
            try:
                percentages = self.segment_image(str(image_path))
                result = {"filename": image_path.name}
                result.update(percentages)
                results.append(result)
                print(f"Processed: {image_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

        return df

    def create_simplified_features(
        self,
        percentages_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create simplified feature set from segmentation results.

        Combines similar categories into meaningful urban features.

        Args:
            percentages_df: DataFrame with raw segmentation percentages

        Returns:
            DataFrame with simplified features
        """
        simplified = percentages_df.copy()

        # Keep filename if it exists
        feature_columns = []

        if 'filename' in simplified.columns:
            feature_columns.append('filename')

        # Map categories to simplified features
        # Note: Actual column names depend on the model's output
        category_mapping = {
            'unlabelled': 'unlabelled',
            'road': 'road',
            'sidewalk': 'green',
            'building': 'building',
            'wall': 'building2',
            'sky': 'sky',
            'person': 'pedestrian',
            'rider': 'pedestrian',
            'ground': 'ground'
        }

        # Create simplified features
        simplified_features = {}

        for new_name in set(category_mapping.values()):
            matching_cols = [k for k, v in category_mapping.items() if v == new_name and k in simplified.columns]
            if matching_cols:
                simplified_features[new_name] = simplified[matching_cols].sum(axis=1)

        # Combine
        result_df = pd.DataFrame(simplified_features)

        if 'filename' in simplified.columns:
            result_df.insert(0, 'filename', simplified['filename'])

        return result_df


def main():
    """Main function to demonstrate usage."""
    segmenter = StreetViewSegmenter()

    # Process existing images
    folders = ["hongdae", "syarosu", "ssook"]

    for folder in folders:
        input_path = f"data/raw/{folder}"
        output_csv = f"data/processed/{folder}_segmentation.csv"

        if os.path.exists(input_path):
            print(f"\nProcessing {folder}...")
            df = segmenter.process_folder(input_path, output_csv)
            print(f"Completed: {len(df)} images processed")
        else:
            print(f"Folder not found: {input_path}")


if __name__ == "__main__":
    main()
