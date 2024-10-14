import time
import re
import csv
import os
import concurrent.futures
from datetime import datetime

from selenium import webdriver
from bs4 import BeautifulSoup

from pokemon import Pokemon
from process_names import process_names

# Names of Pokémon sets as they appear in pricecharting.com URLs
POKEMON_SETS = process_names()
MAX_WORKERS = 10


def pokemon_csv(pokemon_sets: list[Pokemon]):
    """Creates a CSV file to store scraped Pokémon card data.

    Args:
        pokemon_sets: List of Pokémon objects containing scraped data.
    """
    dt = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    directory = dt
    filename = "pokemon_prices.csv"

    if not os.path.exists(directory):
        os.mkdir(directory)

    with open(os.path.join(directory, filename), "w", newline="") as csv_file:
        fieldnames = [
            "pokemon_item",
            "set",
            "ungraded",
            "grade_nine",
            "grade_ten",
            "date(D/M/Y)",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for p in pokemon_sets:
            writer.writerow(
                {
                    "pokemon_item": p.get_title(),
                    "set": p.get_pokemon_set(),
                    "ungraded": p.get_ungraded_price(),
                    "grade_nine": p.get_grade_nine_price(),
                    "grade_ten": p.get_grade_ten_price(),
                    "date(D/M/Y)": dt.split("_")[0].replace(".", "/"),
                }
            )


def scroll_to_bottom(pokemon_set, browser):
    """Scrolls to the bottom of the webpage to load all Pokémon cards.

    Args:
        pokemon_set: Name of the Pokémon set as it appears in the pricecharting URL.
        browser: Selenium WebDriver instance.

    Returns:
        WebDriver: The browser instance after scrolling to the bottom.
    """
    SCROLL_PAUSE_TIME = 1.5
    browser.get(f"https://www.pricecharting.com/console/{pokemon_set}")

    prev_height = browser.execute_script("return document.body.scrollHeight")
    at_bottom = False

    while True:
        time.sleep(SCROLL_PAUSE_TIME)
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        curr_height = browser.execute_script("return document.body.scrollHeight")

        if prev_height == curr_height:
            if at_bottom:
                break
            at_bottom = True
        else:
            at_bottom = False

        prev_height = curr_height

    return browser


def scrape_values(pokemon_set, browser):
    """Scrapes titles and values for each Pokémon card in the webpage.

    Args:
        pokemon_set: Name of the Pokémon set as it appears in the pricecharting URL.
        browser: Selenium WebDriver instance.

    Returns:
        List[Pokemon]: List of Pokémon objects with scraped data.
    """
    pokemon = []
    soup = BeautifulSoup(browser.page_source, "html.parser")

    for each_part in soup.select('tr[id*="product-"]'):
        title = re.search(
            r">(.*?)</a>", str(each_part.select('td[class="title"]'))
        ).group(1)
        ungraded_price = re.findall(
            r"\d+\.\d+", str(each_part.select('td[class="price numeric used_price"]'))
        )
        ungraded_price = ungraded_price[0] if ungraded_price else "N/A"
        grade_nine_price = re.findall(
            r"\d+\.\d+", str(each_part.select('td[class="price numeric cib_price"]'))
        )
        grade_nine_price = grade_nine_price[0] if grade_nine_price else "N/A"
        grade_ten_price = re.findall(
            r"\d+\.\d+", str(each_part.select('td[class="price numeric new_price"]'))
        )
        grade_ten_price = grade_ten_price[0] if grade_ten_price else "N/A"

        new_pokemon = Pokemon(
            title, pokemon_set, ungraded_price, grade_nine_price, grade_ten_price
        )
        pokemon.append(new_pokemon)

    return pokemon


def pull_prices(pokemon_set):  # Removed the browser argument
    """Pulls values from pricecharting.com for a given Pokémon set.

    Args:
        pokemon_set: Name of the Pokémon set.

    Returns:
        List[Pokemon]: List of Pokémon objects with scraped data.
    """
    with webdriver.Firefox() as browser:
        print(f"Pulling values for {pokemon_set} set\n")
        scroll_to_bottom(pokemon_set, browser)
        return scrape_values(pokemon_set, browser)


def main():
    """Executes the program: scrapes values for each Pokémon set and creates a CSV file."""
    all_pokemon = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(pull_prices, pokemon_set) for pokemon_set in POKEMON_SETS
        ]

        for future in concurrent.futures.as_completed(futures):
            pokemon_set = POKEMON_SETS[futures.index(future)]
            try:
                pokemon_items = future.result()
                all_pokemon.extend(pokemon_items)
                print(f"{pokemon_set} Pokémon card set successfully scraped\n")
            except Exception as exc:
                print(f"{pokemon_set} Pokémon card set generated an exception: {exc}")

    pokemon_csv(all_pokemon)


if __name__ == "__main__":
    """Entry point for the program."""
    print("Scraping values from pricecharting.com\n")
    main()
    print("Finished scraping values from pricecharting.com")
