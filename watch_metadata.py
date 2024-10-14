import os
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import ROOT_IMAGE_DIRECTORY, SERVER_URL
from utils import send_directory_to_server

class MetaJsonCreationHandler(FileSystemEventHandler):
    def __init__(self, root_folder, server_url):
        self.root_folder = root_folder
        self.server_url = server_url

    def on_created(self, event):
        if 'meta.json' in event.src_path:
            parent_directory = os.path.dirname(event.src_path)
            print(f"'meta.json' file created in: {parent_directory}")
            send_directory_to_server(parent_directory, self.server_url)

def main():
    path_to_watch = ROOT_IMAGE_DIRECTORY
    server_url = SERVER_URL

    # Setup event handler and observer
    event_handler = MetaJsonCreationHandler(root_folder=path_to_watch, server_url=server_url)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()

    print(f"Watching for creation of 'meta.json' files in '{path_to_watch}'...")

    try:
        while True:
            # Keep the script running
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
