"""
Where the photo training staves are listed (convert_photos.py writes them). Kept apart from
convert_photos.py, so train.py can name the index without importing the engraving libraries.
"""

from pathlib import Path

photos_root = Path(__file__).resolve().parent.parent.parent / "datasets" / "photo_pages"
photos_train_index = str(photos_root / "index.txt")
