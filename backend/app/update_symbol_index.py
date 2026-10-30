"""Download the NASDAQ Trader symbol directory and rebuild the local index.

Usage (from backend/):  python -m app.update_symbol_index
"""

import httpx

from app.assets import SOURCES, build_index
from app.config import get_settings


def main() -> None:
    texts = {}
    for key, url in SOURCES.items():
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
        texts[key] = response.text
    info = build_index(get_settings().data_dir, texts["nasdaqlisted"], texts["otherlisted"])
    print(f"Symbol index written: {info.count} symbols, retrieved {info.retrieved_at}")


if __name__ == "__main__":
    main()
