"""
Image Getter Module

This module provides functionality to fetch street view images using Google Maps API.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOCATIONS, RAW_DATA_DIR


class StreetViewImageGetter:
    """Class for fetching street view images from Google Maps API."""

    def __init__(self, api_key: str, image_size: str = "256x256"):
        """
        Initialize the StreetViewImageGetter.

        Args:
            api_key: Google Maps API key
            image_size: Size of the images to fetch (default: "256x256")
        """
        self.api_key = api_key
        self.image_size = image_size
        self.api_url = "https://maps.googleapis.com/maps/api/streetview"

    def fetch_images_along_path(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        num_points: int,
        output_folder: str,
        headings: List[int] = [0, 90, 180, 270],
        prefix: str = "image"
    ) -> None:
        """
        Fetch street view images along a path from start to end coordinates.

        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Ending latitude
            end_lon: Ending longitude
            num_points: Number of points to sample along the path
            output_folder: Folder to save images
            headings: List of camera headings (default: [0, 90, 180, 270])
            prefix: Prefix for image filenames (default: "image")
        """
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Calculate increments
        lat_increment = (end_lat - start_lat) / num_points
        lon_increment = (end_lon - start_lon) / num_points

        # Fetch images for each point and heading
        for i in range(num_points):
            current_lat = start_lat + lat_increment * i
            current_lon = start_lon + lon_increment * i

            for heading in headings:
                self._fetch_single_image(
                    lat=current_lat,
                    lon=current_lon,
                    heading=heading,
                    output_path=os.path.join(
                        output_folder,
                        f"{prefix}_{i}_heading_{heading}.jpg"
                    )
                )

        print(f"Successfully fetched {num_points * len(headings)} images to {output_folder}")

    def _fetch_single_image(
        self,
        lat: float,
        lon: float,
        heading: int,
        output_path: str,
        pitch: str = "0"
    ) -> bool:
        """
        Fetch a single street view image.

        Args:
            lat: Latitude
            lon: Longitude
            heading: Camera heading (0-360)
            output_path: Path to save the image
            pitch: Camera pitch (default: "0")

        Returns:
            True if successful, False otherwise
        """
        params = {
            "size": self.image_size,
            "location": f"{lat},{lon}",
            "heading": str(heading),
            "pitch": pitch,
            "key": self.api_key
        }

        try:
            response = requests.get(self.api_url, params=params)
            if response.status_code == 200:
                with open(output_path, "wb") as file:
                    file.write(response.content)
                return True
            else:
                print(f"Failed to fetch image: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error fetching image: {e}")
            return False


def main():
    """Main function to demonstrate usage."""
    import dotenv

    # Load environment variables
    dotenv.load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or provide sample data.")
        return

    # Initialize image getter
    getter = StreetViewImageGetter(api_key=api_key)

    # Fetch images for each configured location
    for location_name, config in LOCATIONS.items():
        print(f"\nFetching images for {location_name}...")

        try:
            start_coords: Tuple[float, float] = config["start_coords"]
            end_coords: Tuple[float, float] = config["end_coords"]
            num_points: int = config["num_points"]
        except KeyError as exc:
            missing_key = exc.args[0]
            print(
                f"Skipping {location_name}: configuration missing '{missing_key}'"
            )
            continue

        output_folder = RAW_DATA_DIR / location_name

        getter.fetch_images_along_path(
            start_lat=start_coords[0],
            start_lon=start_coords[1],
            end_lat=end_coords[0],
            end_lon=end_coords[1],
            num_points=num_points,
            output_folder=str(output_folder),
            prefix=f"{location_name}_image"
        )


if __name__ == "__main__":
    main()
