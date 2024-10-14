def process_names(name_file="names.txt"):
    """Process console names from scraper.py to be used in the program

    Returns:
        (list of strings) console names
    """
    with open("names.txt", "r") as f:
        names = f.readlines()
        for line in names:
            line = line.strip()
            line = line.replace("Pokemon", ",Pokemon")
            line = line.lower()
            line = line.replace(" ", "-")
            return line.split(",")

if __name__ == "__main__":
    print(process_names())
    with open("processed_names.txt", "w") as f:
        for name in process_names():
            f.write(name + "\n")
