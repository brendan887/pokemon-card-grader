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
import re

# Global variables
DEFAULT_THREAD_WORKERS: int = 5
GRADES = ["psa 10", "psa 9", "psa 8", "psa 7"]
IMG_PER_GRADE = 2

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
    item: dict, do_send_directory_to_server: bool = False, grade: str = ""
) -> Optional[str]:
    """Processes a single listing, including downloading images and writing metadata."""
    listing_id: str = str(item.get("itemId"))
    title: str = str(item.get("title")).lower()
    condition: str = item.get("condition", {}).get("conditionDisplayName", "").lower()
    price_info = item.get("sellingStatus", {}).get("currentPrice", {})
    price_value = price_info.get("value", "Unknown")
    currency = price_info.get("_currencyId", "USD")
    price: str = f"{price_value} {currency}"
    picture_urls: List[str] = item.get("pictureURLSuperSize", [])
    if isinstance(picture_urls, str):
        picture_urls = [picture_urls]

    # Use regex to check if the expected grade is in the title
    # Create a pattern to match variations like 'psa10', 'psa 10', 'psa-10', etc.
    grade_pattern = rf'\bpsa[-\s]*{grade[-2:]}\b'  # Extract the numeric part of the grade
    if not re.search(grade_pattern, title, re.IGNORECASE):
        print(f"Skipping listing {listing_id}, title does not contain expected grade '{grade}'.")
        return None

    if len(picture_urls) < 1:
        print(f"Skipping listing {listing_id}, no images found.")
        return None
    else:
        print(f"Processing listing {listing_id} with {len(picture_urls)} images")

    directory: str = os.path.join(ROOT_IMAGE_DIRECTORY, grade, listing_id)
    if os.path.exists(directory):
        print(f"Skipping {directory}, already exists.")
        return None

    os.makedirs(directory, exist_ok=True)
    for url in picture_urls:
        download_image(url, directory)

    write_metadata(directory, title, listing_id, len(picture_urls), condition=condition, price=price)

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
        for grade in GRADES:
            full_search_query = f"{search_query} {grade}"
            print(f"Processing search query: {full_search_query}")
            items_downloaded = 0
            page_number = 1
            already_attempted_ids = set()
            MAX_PAGES = 10  # Prevent infinite loop
            while items_downloaded < IMG_PER_GRADE and page_number <= MAX_PAGES:
                response = api.execute(
                    "findItemsAdvanced",
                    {
                        "keywords": full_search_query,
                        "itemFilter": [{"name": "ListingType", "value": "FixedPrice"}],
                        "paginationInput": {"entriesPerPage": "100", "pageNumber": str(page_number)},
                        "outputSelector": [
                            "PictureURLSuperSize",
                            "ConditionDisplayName",
                            "CurrentPrice",
                        ],
                    },
                )
                items = response.dict().get("searchResult", {}).get("item", [])
                if not items:
                    print(f"No more items found for {full_search_query}")
                    break

                # Filter out items we've already attempted
                new_items = []
                for item in items:
                    item_id = str(item.get("itemId"))
                    if item_id not in already_attempted_ids:
                        new_items.append(item)
                        already_attempted_ids.add(item_id)
                if not new_items:
                    print(f"No new items found on page {page_number} for {full_search_query}")
                    page_number += 1
                    continue

                # Limit the items to process to the number of images we still need
                items_needed = IMG_PER_GRADE - items_downloaded
                items_to_process = new_items[:items_needed]

                with ThreadPoolExecutor(max_workers=max_thread_workers) as executor:
                    futures = [executor.submit(process_listing, item, do_send_directory_to_server, grade) for item in items_to_process]
                    for future in as_completed(futures):
                        try:
                            processed_id = future.result()
                            if processed_id:
                                processed_ids.add(processed_id)
                                items_downloaded += 1
                                if items_downloaded >= IMG_PER_GRADE:
                                    break
                        except Exception as exc:
                            print(f"Listing processing generated an exception: {exc}")

                if items_downloaded >= IMG_PER_GRADE:
                    break
                page_number += 1
    except Exception as e:
        print(str(e))
    return processed_ids



def delete_unused_directories(all_processed_ids: Set[str]):
    """Deletes directories for listing IDs not found in the current run."""
    for grade in GRADES:
        grade_directory = os.path.join(ROOT_IMAGE_DIRECTORY, grade)
        if not os.path.exists(grade_directory):
            continue
        for listing_id in os.listdir(grade_directory):
            directory = os.path.join(grade_directory, listing_id)
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
