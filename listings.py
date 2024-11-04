import argparse
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from ebaysdk.finding import Connection as Finding
from typing import List, Set, Optional
from multiprocessing import cpu_count
from config import ROOT_IMAGE_DIRECTORY, EBAY_APP_ID, SEARCH_QUERIES, SERVER_URL
from utils import write_metadata, send_directory_to_server
from dotenv import load_dotenv

# Global variables
DEFAULT_THREAD_WORKERS: int = 5


def download_image(url: str, directory: str) -> None:
    """Downloads an image from a URL into a specified directory."""
    filename: str = url.split("/")[-1]
    path: str = os.path.join(directory, filename)

    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {path}")
    else:
        print(f"Error downloading {url}")


def process_listing(
    item: dict, do_send_directory_to_server: bool = False
) -> Optional[str]:
    """Processes a single listing, including downloading images and writing metadata."""
    print(item)
    listing_id: str = str(item.get("itemId"))
    title: str = str(item.get("title"))
    condition: str = item.get("condition", {}).get("conditionDisplayName", "Unknown")
    price: str = (
        item.get("sellingStatus", {}).get("currentPrice", {}).get("value", "Unknown")
        + " "
        + item.get("sellingStatus", {})
        .get("currentPrice", {})
        .get("_currencyId", "USD")
    )
    picture_urls: List[str] = item.get("pictureURLSuperSize", [])
    if isinstance(picture_urls, str):
        picture_urls = [picture_urls]

    if len(picture_urls) < 1:
        print(f"Skipping listing {listing_id}, {len(picture_urls)} images found.")
        return None
    else:
        print(f"Processing listing {listing_id} with {len(picture_urls)} images")

    directory: str = os.path.join(ROOT_IMAGE_DIRECTORY, listing_id)
    if os.path.exists(directory):
        print(f"Skipping {directory}, already exists.")
        return None

    os.makedirs(directory, exist_ok=True)
    for url in picture_urls:
        download_image(url, directory)

    write_metadata(directory, title, listing_id, len(picture_urls), condition, price)

    if do_send_directory_to_server:
        send_directory_to_server(directory, SERVER_URL)

    return listing_id


def process_search_query(
    search_query: str,
    do_send_directory_to_server: bool = False,
    max_thread_workers: int = DEFAULT_THREAD_WORKERS,
) -> Set[str]:
    """Processes a single search query and returns a set of processed listing IDs."""
    processed_ids: Set[str] = set()
    try:
        api = Finding(appid=EBAY_APP_ID, config_file=None)
        response = api.execute(
            "findItemsAdvanced",
            {
                "keywords": search_query,
                "itemFilter": [{"name": "ListingType", "value": "FixedPrice"}],
                "paginationInput": {"entriesPerPage": "100"},
                "outputSelector": [
                    "PictureURLSuperSize",
                    "ConditionDisplayName",
                    "CurrentPrice",
                ],
            },
        )
        items = response.dict().get("searchResult", {}).get("item", [])
        with ThreadPoolExecutor(max_workers=max_thread_workers) as executor:
            future_to_item = {
                executor.submit(
                    process_listing, item, do_send_directory_to_server
                ): item
                for item in items
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    processed_id = future.result()
                    if processed_id:
                        processed_ids.add(processed_id)
                except Exception as exc:
                    print(f"Listing processing generated an exception: {exc}")
    except Exception as e:
        print(str(e))
    return processed_ids


def delete_unused_directories(all_processed_ids: Set[str]):
    """Deletes directories for listing IDs not found in the current run."""
    for listing_id in os.listdir(ROOT_IMAGE_DIRECTORY):
        directory = os.path.join(ROOT_IMAGE_DIRECTORY, listing_id)
        if listing_id not in all_processed_ids and os.path.isdir(directory):
            os.rmdir(directory)
            print(f"Deleted unused directory: {directory}")


def main(arguments):
    """Processes multiple search queries in parallel using different processes and handles cleanup."""
    while True:
        all_processed_ids = set()
        with ProcessPoolExecutor(max_workers=arguments.process_workers) as executor:
            futures = [
                executor.submit(
                    process_search_query,
                    query,
                    arguments.do_send_directory_to_server,
                    arguments.thread_workers,
                )
                for query in arguments.search_queries
            ]
            for future in as_completed(futures):
                try:
                    processed_ids = future.result()
                    all_processed_ids.update(processed_ids)
                except Exception as exc:
                    print(f"Search query processing generated an exception: {exc}")

        delete_unused_directories(all_processed_ids)
        time.sleep(arguments.delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download eBay listing images.")
    parser.add_argument(
        "--search-queries",
        nargs="+",
        required=False,
        default=SEARCH_QUERIES,
        help="Search queries to process.",
    )
    parser.add_argument(
        "--do-send-directory-to-server",
        action="store_true",
        default=False,
        help="Send directory to server after processing.",
    )
    parser.add_argument(
        "--thread-workers",
        type=int,
        default=DEFAULT_THREAD_WORKERS,
        help="Number of thread workers for handling listings.",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=cpu_count(),
        help="Number of process workers for search queries.",
    )
    parser.add_argument(
        "--delay", type=int, default=3600, help="Delay between iterations in seconds."
    )

    args = parser.parse_args()
    main(args)
