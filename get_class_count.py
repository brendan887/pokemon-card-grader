import os

# Define the root directory
ROOT_IMAGE_DIRECTORY = 'listing_images'

# List of grades to check
GRADES = ['psa 10', 'psa 9', 'psa 8', 'psa 7']

def count_folders_in_directory(directory):
    """Counts the number of folders in a given directory."""
    try:
        # Get the list of items in the directory
        items = os.listdir(directory)
        # Filter the list to include only directories
        folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
        return len(folders)
    except FileNotFoundError:
        print(f"Directory '{directory}' does not exist.")
        return 0

def main():
    total = 0
    for grade in GRADES:
        directory = os.path.join(ROOT_IMAGE_DIRECTORY, grade)
        folder_count = count_folders_in_directory(directory)
        print(f"There are {folder_count} folders in '{directory}'.")
        total += folder_count
    print("Total:", total)
        

if __name__ == '__main__':
    main()
