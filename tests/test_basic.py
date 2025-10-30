"""
Basic tests for Seoul Street View Analysis project.

Run with: python -m pytest tests/
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all main modules can be imported."""
    try:
        import src.seoul_streetview.data.image_getter
        import src.seoul_streetview.data.sample_generator
        import src.seoul_streetview.modeling.modeling
        import src.seoul_streetview.segmentation.segmenter
        from src.seoul_streetview.utils import setup_logging
        from config import settings
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_config_settings():
    """Test configuration settings."""
    from config.settings import LOCATIONS, IMAGE_SETTINGS, FEATURE_NAMES

    # Test locations
    assert len(LOCATIONS) >= 3, "Should have at least 3 locations"
    assert "hongdae" in LOCATIONS, "Should have hongdae location"
    assert "syarosu" in LOCATIONS, "Should have syarosu location"

    required_keys = {"start_coords", "end_coords", "num_points"}
    for name, location in LOCATIONS.items():
        missing = required_keys - location.keys()
        assert not missing, f"Location '{name}' missing keys: {missing}"

    # Test image settings
    assert IMAGE_SETTINGS["size"] == "256x256", "Image size should be 256x256"
    assert len(IMAGE_SETTINGS["headings"]) == 4, "Should have 4 headings"

    # Test feature names
    assert len(FEATURE_NAMES) >= 5, "Should have at least 5 features"
    assert "building" in FEATURE_NAMES, "Should have building feature"


def test_data_files_exist():
    """Test that required data files exist."""
    import os

    required_files = [
        "class_percentages.csv",
        "Urban_vitality_index.xlsx"
    ]

    for file in required_files:
        assert os.path.exists(file), f"Required file {file} not found"


def test_directory_structure():
    """Test that required directories exist."""
    from config.settings import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

    assert DATA_DIR.exists(), "Data directory should exist"
    assert RAW_DATA_DIR.exists(), "Raw data directory should exist"
    assert PROCESSED_DATA_DIR.exists(), "Processed data directory should exist"


def test_utils():
    """Test utility functions."""
    from src.seoul_streetview.utils import ensure_dir, get_project_root, format_percentage

    # Test ensure_dir
    test_dir = ensure_dir(Path("data/test"))
    assert test_dir.exists(), "ensure_dir should create directory"

    # Test get_project_root
    root = get_project_root()
    assert root.exists(), "Project root should exist"
    assert (root / "README.md").exists(), "README should exist in project root"

    # Test format_percentage
    assert format_percentage(50.123, 2) == "50.12%"
    assert format_percentage(100.0, 1) == "100.0%"


if __name__ == "__main__":
    """Run tests manually without pytest."""
    print("Running basic tests...\n")

    tests = [
        ("Imports", test_imports),
        ("Config Settings", test_config_settings),
        ("Data Files", test_data_files_exist),
        ("Directory Structure", test_directory_structure),
        ("Utils", test_utils)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}: PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name}: ERROR - {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
