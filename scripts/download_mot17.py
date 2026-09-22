import sys
import urllib.request
import zipfile
from pathlib import Path


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    # handle cases where total_size is unknown (-1)
    if total_size > 0:
        percent = downloaded / total_size * 100
        print(f"\rDownloading MOT17... {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="", flush=True)
    else:
        print(f"\rDownloading MOT17... {downloaded / (1024*1024):.1f} MB downloaded", end="", flush=True)

url = "https://motchallenge.net/data/MOT17.zip"
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
zip_path = data_dir / "MOT17.zip"
extract_dir = data_dir

if not zip_path.exists():
    print(f"Downloading {url} to {zip_path}...", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', -1))
            downloaded = 0
            block_size = 8192
            with open(zip_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    download_progress(downloaded // block_size, block_size, total_size)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)
else:
    print(f"{zip_path} already exists, skipping download.", flush=True)

print("Extracting ZIP file...", flush=True)
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Done extracting MOT17 dataset.", flush=True)
except Exception as e:
    print(f"Extraction failed: {e}")
    sys.exit(1)
