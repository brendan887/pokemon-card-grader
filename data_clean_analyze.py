import re
import requests

def get_pokemon_names():
    pokemon_names = []
    url = 'https://pokeapi.co/api/v2/pokemon?limit=10000'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for pokemon in data['results']:
            pokemon_names.append(pokemon['name'].lower())
    else:
        print("Failed to retrieve Pokémon names.")
    return pokemon_names


def standardize_card_name(line, pokemon_names):
    original_line = line.strip()
    categories = {
        'trainer': False,
        'pokemon': False,
        'full_art': False,
        'half_art': False,
        'gx': False,
        'ex': False,
        'v': False,
        'vmax': False
    }

    if re.search(r'Full Art', original_line, re.IGNORECASE):
        categories['full_art'] = True
    else:
        categories['half_art'] = True

    if re.search(r'\bGX\b', original_line, re.IGNORECASE):
        categories['gx'] = True
    if re.search(r'\bEX\b', original_line, re.IGNORECASE):
        categories['ex'] = True
    if re.search(r'\bV\b', original_line, re.IGNORECASE):
        categories['v'] = True
    if re.search(r'\bVMAX\b', original_line, re.IGNORECASE):
        categories['vmax'] = True

    card_name = original_line.split('-')[0]  # Get the part before the dash
    card_name_tokens = re.findall(r'\b\w+\b', card_name.lower())

    if any(token in pokemon_names for token in card_name_tokens):
        categories['pokemon'] = True
    else:
        categories['trainer'] = True

    line = re.sub(r'\s*\(.*?\)\s*', ' ', line)

    if '-' in line:
        name_part, numbering_part = line.split('-', 1)
        name_part = name_part.strip()
        numbering_part = numbering_part.strip()
    else:
        name_part = line.strip()
        numbering_part = ''

    tokens = name_part.split()
    uppercase_tokens = [token for token in tokens if token.isupper()]
    other_tokens = [token for token in tokens if not token.isupper()]
    rearranged_name = ' '.join(other_tokens + uppercase_tokens)

    if numbering_part:
        standardized_line = f"{rearranged_name} {numbering_part}"
    else:
        standardized_line = rearranged_name

    standardized_line = standardized_line.lower()

    return standardized_line, categories

def create_card_list(file_path, pokemon_names):
    card_list = []
    counts = {
        'trainer': 0,
        'pokemon': 0,
        'full_art': 0,
        'half_art': 0,
        'gx': 0,
        'ex': 0,
        'v': 0,
        'vmax': 0
    }
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            standardized_line, categories = standardize_card_name(line, pokemon_names)
            card_list.append(standardized_line)

            for category in counts:
                if categories[category]:
                    counts[category] += 1

    return card_list, counts

def print_list_as_code(card_list):
    print('pokemon_cards = [')
    for card in card_list:
        print(f'    "{card}",')
    print(']')

def print_counts(counts):
    print("\nCategory Counts:")
    for category, count in counts.items():
        print(f"{category.capitalize()}: {count}")

pokemon_names = get_pokemon_names()

card_list, counts = create_card_list('cards_list.txt', pokemon_names)
print_list_as_code(card_list)
print_counts(counts)