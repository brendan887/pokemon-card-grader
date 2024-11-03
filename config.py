from typing import List
from dotenv import load_dotenv
import os

ROOT_IMAGE_DIRECTORY: str = "listing_images"
SERVER_URL: str = "http://localhost:8000/notify"
EBAY_APP_ID: str = os.getenv("EBAY_APP_ID")
SEARCH_QUERIES: List[str] = [
    "giratina v 186",
    # "giratina v 186 psa 10",
    # "giratina v 186 psa 9",
    # "giratina v 186 psa 8",
    # "giratina v 186 psa 7",
    # "charizard ex 105",
    # "charizard ex 105 psa 10",
    # "charizard ex 105 psa 9",
    # "charizard ex 105 psa 8",
    # "charizard ex 105 psa 7",
    # "rayquaza gold star 107",
    # "rayquaza gold star 107 psa 10",
    # "rayquaza gold star 107 psa 9",
    # "rayquaza gold star 107 psa 8",
    # "rayquaza gold star 107 psa 7",
]
