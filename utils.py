import json
import os
import requests


def write_metadata(
    directory: str,
    listing_name: str,
    listing_id: str,
    number_of_images: int,
    condition: str,
    price: str,
) -> None:
    """Writes metadata to a JSON file."""
    metadata: dict = {
        "listing_name": listing_name,
        "listing_id": listing_id,
        "number_of_images": number_of_images,
        "condition": condition,
        "price": price,
    }
    path: str = os.path.join(directory, "meta.json")
    with open(path, "w") as f:
        json.dump(metadata, f)


def write_progress(
    directory: str,
    cropped: bool = False,
    categorized_front_back: bool = False,
    rating: float = None,
) -> None:
    """Writes progress to a JSON file."""
    progress: dict = {
        "cropped": cropped,
        "categorized_front_back": categorized_front_back,
    }
    if rating is not None:
        progress["rating"] = rating
    path: str = os.path.join(directory, "meta.json")
    with open(path, "w") as f:
        json.dump(progress, f)


def send_directory_to_server(directory: str, server_url: str) -> None:
    """Sends a directory to a server."""
    try:
        response = requests.post(server_url, json={"directory": directory})
        if response.status_code == 200:
            print(f"Successfully notified server about change in: {directory}")
        else:
            print(f"Failed to notify server. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
