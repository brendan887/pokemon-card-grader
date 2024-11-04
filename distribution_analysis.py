from config import SEARCH_QUERIES

def get_pokemon_count(card_list):
    pokemon_count, trainer_count = 0, 0
    return pokemon_count, trainer_count

def get_full_art_count(card_list):
    full_art_count, half_art_count = 0, 0
    return full_art_count, half_art_count

print(get_pokemon_count(SEARCH_QUERIES))
print(get_full_art_count(SEARCH_QUERIES))
