"""
Custom exceptions for Seoul Street View Analysis project.
"""


class SeoulStreetViewError(Exception):
    """Base exception for Seoul Street View Analysis."""
    pass


class APIKeyError(SeoulStreetViewError):
    """Raised when Google Maps API key is missing or invalid."""
    pass


class ImageFetchError(SeoulStreetViewError):
    """Raised when image fetching fails."""
    pass


class SegmentationError(SeoulStreetViewError):
    """Raised when image segmentation fails."""
    pass


class ModelTrainingError(SeoulStreetViewError):
    """Raised when model training fails."""
    pass


class DataError(SeoulStreetViewError):
    """Raised when data is invalid or missing."""
    pass


class ConfigurationError(SeoulStreetViewError):
    """Raised when configuration is invalid."""
    pass
