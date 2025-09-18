import re
import json
import requests
import logging
import urllib.parse
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional
from tqdm import tqdm
import time
import argparse
from fetch_all_models import fetch_all_models
import sys

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "civitAI_Model_downloader.txt")
OUTPUT_DIR = "model_downloads"
MAX_PATH_LENGTH = 200
VALID_DOWNLOAD_TYPES = ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', 'All']
BASE_URL = "https://civitai.com/api/v1/models"

logger_md = logging.getLogger('md')
logger_md.setLevel(logging.DEBUG)
file_handler_md = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler_md.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_md.setFormatter(formatter)
logger_md.addHandler(file_handler_md)

# Function to sanitize directory names
def sanitize_directory_name(name):
    return name.rstrip()  # Remove trailing whitespace characters


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Download model files and images from Civitai API.")
    parser.add_argument("usernames", nargs='+', type=str, help="Enter one or more usernames you want to download from.")
    parser.add_argument("--retry_delay", type=int, default=10, help="Retry delay in seconds.")
    parser.add_argument("--max_tries", type=int, default=3, help="Maximum number of retries.")
    parser.add_argument("--max_threads", type=int, default=5, help="Maximum number of concurrent threads. Too many produces API Failure.")
    parser.add_argument("--token", type=str, default=None, help="API Token for Civitai.")
    parser.add_argument("--download_type", type=str, default=None, help="Specify the type of content to download: 'Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', or 'All' (default).")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory where downloaded models should be stored.")
    return parser


def validate_download_type(download_type: Optional[str]) -> str:
    if download_type is None:
        return 'All'
    if download_type not in VALID_DOWNLOAD_TYPES:
        raise ValueError(
            "Invalid download type specified. Valid types are: 'Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', or 'All'."
        )
    return download_type


def _clean_username(username: str) -> Optional[str]:
    username = username.strip()
    return username or None


@dataclass
class DownloaderConfig:
    usernames: List[str] = field(default_factory=list)
    retry_delay: int = 10
    max_tries: int = 3
    max_threads: int = 5
    token: Optional[str] = None
    download_type: str = 'All'
    output_dir: str = field(default=OUTPUT_DIR)

    def __post_init__(self):
        cleaned_usernames = [_clean_username(user) for user in self.usernames]
        self.usernames = [user for user in cleaned_usernames if user]
        if not self.usernames:
            raise ValueError("At least one username must be provided.")

        self.download_type = validate_download_type(self.download_type)
        self.output_dir = sanitize_directory_name(self.output_dir)


def create_config_from_args(args: argparse.Namespace) -> DownloaderConfig:
    return DownloaderConfig(
        usernames=args.usernames,
        retry_delay=args.retry_delay,
        max_tries=args.max_tries,
        max_threads=args.max_threads,
        token=args.token,
        download_type=args.download_type,
        output_dir=args.output_dir,
    )


def ensure_output_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def emit(message: str, log_callback: Optional[Callable[[str], None]] = None, level: int = logging.INFO) -> None:
    logger_md.log(level, message)
    if log_callback:
        log_callback(message)
    else:
        print(message)

def read_summary_data(username: str, log_callback: Optional[Callable[[str], None]] = None):
    """Read summary data from a file."""
    summary_path = os.path.join(SCRIPT_DIR, f"{username}.txt")
    data = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as file:
            for line in file:
                if 'Total - Count:' in line:
                    total_count = int(line.strip().split(':')[1].strip())
                    data['Total'] = total_count
                elif ' - Count:' in line:
                    category, count = line.strip().split(' - Count:')
                    data[category.strip()] = int(count.strip())
    except FileNotFoundError:
        emit(f"File {summary_path} not found.", log_callback, level=logging.WARNING)
    return data

def sanitize_name(name, folder_name=None, max_length=MAX_PATH_LENGTH, subfolder=None, output_dir=None, username=None):
    """Sanitize a name for use as a file or folder name."""
    base_name, extension = os.path.splitext(name)

    if folder_name and base_name == folder_name:
        return name

    if folder_name:
        base_name = base_name.replace(folder_name, "").strip("_")

    # Remove problematic characters and control characters
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)

    # Handle reserved names (Windows specific)
    reserved_names = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if base_name.upper() in reserved_names:
        base_name = '_'

    # Reduce multiple underscores to single and trim leading/trailing underscores and dots
    base_name = re.sub(r'__+', '_', base_name).strip('_.')
    
    # Calculate max length of base name considering the path length
    if subfolder and output_dir and username:
        path_length = len(os.path.join(output_dir, username, subfolder))
        max_base_length = max_length - len(extension) - path_length
        base_name = base_name[:max_base_length].rsplit('_', 1)[0]

    sanitized_name = base_name + extension
    return sanitized_name.strip()


def download_file_or_image(
    url: str,
    output_path: str,
    session: requests.Session,
    config: DownloaderConfig,
    username: str,
    retry_count: int = 0,
    log_callback: Optional[Callable[[str], None]] = None,
):
    """Download a file or image from the provided URL."""
    if os.path.exists(output_path):
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    progress_bar = None
    try:
        response = session.get(url, stream=True, timeout=(20, 40))
        if response.status_code == 404:
            emit(f"File not found: {url}", log_callback, level=logging.WARNING)
            return False
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            leave=False,
            disable=log_callback is not None,
        )
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        progress_bar.close()
        if output_path.endswith('.safetensor') and os.path.getsize(output_path) < 4 * 1024 * 1024:  # 4MB
            if retry_count < config.max_tries:
                emit(
                    f"File {output_path} is smaller than expected. Try to download again (attempt {retry_count + 1}).",
                    log_callback,
                    level=logging.WARNING,
                )
                time.sleep(config.retry_delay)
                return download_file_or_image(
                    url,
                    output_path,
                    session,
                    config,
                    username,
                    retry_count + 1,
                    log_callback,
                )
            download_errors_log = os.path.join(SCRIPT_DIR, f'{username}.download_errors.log')
            with open(download_errors_log, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Failed to download {url} after {config.max_tries} attempts.\n")
            return False
        return True
    except (requests.RequestException, TimeoutError, ConnectionResetError) as e:
        if progress_bar:
            progress_bar.close()
        if retry_count < config.max_tries:
            emit(
                f"Error downloading {url}: {e}. Retrying in {config.retry_delay} seconds (attempt {retry_count + 1}).",
                log_callback,
                level=logging.WARNING,
            )
            time.sleep(config.retry_delay)
            return download_file_or_image(
                url,
                output_path,
                session,
                config,
                username,
                retry_count + 1,
                log_callback,
            )
        download_errors_log = os.path.join(SCRIPT_DIR, f'{username}.download_errors.log')
        with open(download_errors_log, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Failed to download {url} after {config.max_tries} attempts. Error: {e}\n")
        return False
    except Exception as e:  # pylint: disable=broad-except
        if progress_bar:
            progress_bar.close()
        if retry_count < config.max_tries:
            emit(
                f"Error during download: {url}, attempt {retry_count + 1}",
                log_callback,
                level=logging.WARNING,
            )
            time.sleep(config.retry_delay)
            return download_file_or_image(
                url,
                output_path,
                session,
                config,
                username,
                retry_count + 1,
                log_callback,
            )
        download_errors_log = os.path.join(SCRIPT_DIR, f'{username}.download_errors.log')
        with open(download_errors_log, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Error downloading file {output_path} from URL {url}: {e} after {config.max_tries} attempts\n")
        return False
    return True

def download_model_files(
    item_name: str,
    model_version: dict,
    item: dict,
    download_type: str,
    failed_downloads_file: str,
    session: requests.Session,
    config: DownloaderConfig,
    username: str,
    log_callback: Optional[Callable[[str], None]] = None,
):
    """Download related image and model files for each model version."""
    files = model_version.get('files', [])
    images = model_version.get('images', [])
    downloaded = False
    model_id = item['id']
    model_url = f"https://civitai.com/models/{model_id}"
    item_name_sanitized = sanitize_name(item_name, max_length=MAX_PATH_LENGTH)
    model_images = {}
    item_dir = None

    # Extract the description and baseModel
    description = item.get('description') or ''
    base_model = item.get('baseModel')
    trigger_words =  model_version.get('trainedWords', [])
    

    subfolder = 'Other'
    for file in files:
        file_name = file.get('name', '')
        file_url = file.get('downloadUrl', '')

        # Determine subfolder (existing logic)
        if file_name.endswith('.zip'):
            if 'type' in item and item['type'] == 'LORA':
                subfolder = 'Lora'
            elif 'type' in item and item['type'] == 'Training_Data':
                subfolder = 'Training_Data'
            else:
                subfolder = 'Other'
        elif file_name.endswith('.safetensors'):
            if 'type' in item:
                if item['type'] == 'Checkpoint':
                    subfolder = 'Checkpoints'
                elif item['type'] == 'TextualInversion':
                    subfolder = 'Embeddings'
                elif item['type'] in ['VAE', 'LoCon']:
                    subfolder = 'Other'
                else:
                    subfolder = 'Lora'
            else:
                subfolder = 'Lora'
        elif file_name.endswith('.pt'):
            if 'type' in item and item['type'] == 'TextualInversion':
                subfolder = 'Embeddings'
            else:
                subfolder = 'Other'
        else:
            subfolder = 'Other'

        if download_type != 'All' and download_type != subfolder:
            continue

        # Create folder structure
        if base_model:
            item_dir = os.path.join(config.output_dir, username, subfolder, base_model, item_name_sanitized)
            logging.info(f"Using baseModel folder structure for {item_name}: {base_model}")
        else:
            item_dir = os.path.join(config.output_dir, username, subfolder, item_name_sanitized)
            logging.info(f"No baseModel found for {item_name}, using standard folder structure")

        try:
            os.makedirs(item_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directory for {item_name}: {str(e)}")
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name}\n")
                f.write(f"Model URL: {model_url}\n")
                f.write("---\n")
            return item_name, False, model_images

        # Create and write to the description file
        description_file = os.path.join(item_dir, "description.html")
        with open(description_file, "w", encoding='utf-8') as f:
            f.write(description)


        trigger_words_file = os.path.join(item_dir, "triggerWords.txt")
        with open(trigger_words_file, "w", encoding='utf-8') as f:
            for word in trigger_words:
                f.write(f"{word}\n")

        if config.token:
            if '?' in file_url:
                file_url += f"&token={config.token}&nsfw=true"
            else:
                file_url += f"?token={config.token}&nsfw=true"
        else:
            if '?' in file_url:
                file_url += "&nsfw=true"
            else:
                file_url += "?nsfw=true"

        file_name_sanitized = sanitize_name(
            file_name,
            item_name,
            max_length=MAX_PATH_LENGTH,
            subfolder=subfolder,
            output_dir=config.output_dir,
            username=username,
        )
        file_path = os.path.join(item_dir, file_name_sanitized)

        if not file_name or not file_url:
            emit(f"Invalid file entry: {file}", log_callback, level=logging.WARNING)
            continue

        success = download_file_or_image(file_url, file_path, session, config, username, log_callback=log_callback)
        if success:
            downloaded = True
        else:
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name}\n")
                f.write(f"File URL: {file_url}\n")
                f.write("---\n")

        details_file = sanitize_directory_name(os.path.join(item_dir, "details.txt"))
        with open(details_file, "a", encoding='utf-8') as f:
            f.write(f"Model URL: {model_url}\n")
            f.write(f"File Name: {file_name}\n")
            f.write(f"File URL: {file_url}\n")

    if item_dir is not None:
        for image in images:
            image_id = image.get('id', '')
            image_url = image.get('url', '')

            reference_file = file_name if files else "model"
            image_filename_raw = f"{item_name}_{image_id}_for_{reference_file}.jpeg"
            image_filename_sanitized = sanitize_name(
                image_filename_raw,
                item_name,
                max_length=MAX_PATH_LENGTH,
                subfolder=subfolder,
                output_dir=config.output_dir,
                username=username,
            )
            image_path = os.path.join(item_dir, image_filename_sanitized)
            if not image_id or not image_url:
                emit(f"Invalid image entry: {image}", log_callback, level=logging.WARNING)
                continue

            success = download_file_or_image(image_url, image_path, session, config, username, log_callback=log_callback)
            if success:
                downloaded = True
            else:
                with open(failed_downloads_file, "a", encoding='utf-8') as f:
                    f.write(f"Item Name: {item_name}\n")
                    f.write(f"Image URL: {image_url}\n")
                    f.write("---\n")

            details_file = sanitize_directory_name(os.path.join(item_dir, "details.txt"))
            with open(details_file, "a", encoding='utf-8') as f:
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Image URL: {image_url}\n")

    return item_name, downloaded, model_images

def process_username(
    username: str,
    config: DownloaderConfig,
    session: requests.Session,
    log_callback: Optional[Callable[[str], None]] = None,
):
    """Process a username and download the specified type of content."""
    emit(f"Processing username: {username}, Download type: {config.download_type}", log_callback)
    fetch_all_models(config.token, username)
    summary_data = read_summary_data(username, log_callback)
    total_items = summary_data.get('Total', 0)

    if config.download_type == 'All':
        selected_type_count = total_items
        intentionally_skipped = 0
    else:
        selected_type_count = summary_data.get(config.download_type, 0)
        intentionally_skipped = total_items - selected_type_count

    params = {
        "username": username,
    }
    if config.token:
        params["token"] = config.token
    params["nsfw"] = "true"
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    headers = {
        "Content-Type": "application/json"
    }

    failed_downloads_file = os.path.join(SCRIPT_DIR, f"failed_downloads_{username}.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write(f"Failed Downloads for Username: {username}\n\n")

    initial_url = url
    next_page = url
    first_next_page = None

    while True:
        if next_page is None:
            emit("End of pagination reached: 'next_page' is None.", log_callback)
            break

        retry_count = 0
        max_retries = config.max_tries
        retry_delay = config.retry_delay

        while retry_count < max_retries:
            try:
                response = session.get(next_page, headers=headers)
                response.raise_for_status()
                data = response.json()
                break  # Exit retry loop on successful response
            except (requests.RequestException, TimeoutError, json.JSONDecodeError) as e:
                emit(f"Error making API request or decoding JSON response: {e}", log_callback, level=logging.WARNING)
                retry_count += 1
                if retry_count < max_retries:
                    emit(f"Retrying in {retry_delay} seconds...", log_callback, level=logging.INFO)
                    time.sleep(retry_delay)
                else:
                    emit("Maximum retries exceeded. Exiting.", log_callback, level=logging.ERROR)
                    return {
                        "username": username,
                        "downloaded": 0,
                        "failed": selected_type_count,
                        "skipped": intentionally_skipped,
                        "total": total_items,
                    }

        items = data['items']
        metadata = data.get('metadata', {})
        next_page = metadata.get('nextPage')

        if not metadata and not items:
            emit("Termination condition met: 'metadata' is empty.", log_callback)
            break

        if first_next_page is None:
            first_next_page = next_page

        executor = ThreadPoolExecutor(max_workers=config.max_threads)
        download_futures = []
        downloaded_item_names = set()

        for item in items:
            item_name = item['name']
            model_versions = item['modelVersions']
            if item_name in downloaded_item_names:
                continue
            downloaded_item_names.add(item_name)

            for version in model_versions:
                # Include baseModel in the item dictionary
                item_with_base_model = item.copy()
                item_with_base_model['baseModel'] = version.get('baseModel')
                
                future = executor.submit(
                    download_model_files,
                    item_name,
                    version,
                    item_with_base_model,
                    config.download_type,
                    failed_downloads_file,
                    session,
                    config,
                    username,
                    log_callback,
                )
                download_futures.append(future)

        for future in tqdm(
            download_futures,
            desc="Downloading Files",
            unit="file",
            leave=False,
            disable=log_callback is not None,
        ):
            future.result()

        executor.shutdown()

    user_output_dir = os.path.join(config.output_dir, username)
    if config.download_type == 'All':
        downloaded_count = sum(
            len(os.listdir(os.path.join(user_output_dir, category)))
            for category in ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other']
            if os.path.exists(os.path.join(user_output_dir, category))
        )
    else:
        download_dir = os.path.join(user_output_dir, config.download_type)
        downloaded_count = len(os.listdir(download_dir)) if os.path.exists(download_dir) else 0

    failed_count = selected_type_count - downloaded_count

    emit(f"Total items for username {username}: {total_items}", log_callback)
    emit(f"Downloaded items for username {username}: {downloaded_count}", log_callback)
    emit(f"Intentionally skipped items for username {username}: {intentionally_skipped}", log_callback)
    emit(f"Failed items for username {username}: {failed_count}", log_callback)

    return {
        "username": username,
        "downloaded": downloaded_count,
        "failed": failed_count,
        "skipped": intentionally_skipped,
        "total": total_items,
    }


def run_downloader(
    config: DownloaderConfig,
    log_callback: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Execute downloads for all usernames defined in the configuration."""
    ensure_output_directory(config.output_dir)
    results = []
    session = requests.Session()
    for username in config.usernames:
        result = process_username(username, config, session, log_callback)
        results.append(result)
    return results


def prompt_for_token_if_missing(token: Optional[str]) -> Optional[str]:
    if token:
        return token
    return input("Please enter your Civitai API token: ").strip() or None


def main(argv: Optional[Iterable[str]] = None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    args.token = prompt_for_token_if_missing(args.token)
    config = create_config_from_args(args)
    run_downloader(config)


if __name__ == "__main__":
    main()
