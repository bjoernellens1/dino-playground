import os
import urllib.request
import zipfile
import sys

DATA_DIR = "data/coco"
URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print() # Newline

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for name, url in URLS.items():
        zip_name = f"{name}.zip"
        zip_path = os.path.join(DATA_DIR, zip_name)
        
        # Download
        try:
            download_file(url, zip_path)
            
            # Extract
            extract_zip(zip_path, DATA_DIR)
            
            # Optional: Remove zip
            # os.remove(zip_path)
        except Exception as e:
            print(f"Failed to process {name}: {e}")

if __name__ == "__main__":
    main()
