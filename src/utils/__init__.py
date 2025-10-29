"""
Utility functions for Seoul Street View Analysis project.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    from config.settings import LOGGING_CONFIG

    if log_file:
        LOGGING_CONFIG["handlers"]["file"]["filename"] = log_file

    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)


def ensure_dir(directory: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Path to directory

    Returns:
        Path object
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return PROJECT_ROOT


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a percentage value.

    Args:
        value: Percentage value (0-100)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def print_section_header(title: str, width: int = 80, char: str = "=") -> None:
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Total width of header
        char: Character to use for border
    """
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """
    Print progress bar.

    Args:
        current: Current iteration
        total: Total iterations
        prefix: Prefix string
    """
    percentage = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    print(f'\r{prefix}: |{bar}| {percentage:.1f}% Complete', end='')
    if current == total:
        print()


class ProgressTracker:
    """Context manager for tracking progress."""

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Description of the task
        """
        self.total = total
        self.current = 0
        self.description = description

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.current < self.total:
            print()

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items processed
        """
        self.current += n
        print_progress(self.current, self.total, self.description)
