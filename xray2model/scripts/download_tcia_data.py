# Placeholder script for downloading data from The Cancer Imaging Archive (TCIA)
# using their REST API.

# TODO:
# - Obtain a TCIA API Key if required for access/download.
# - Consult the TCIA REST API documentation for exact endpoints, parameters, and response formats:
#   https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guides
# - Implement robust error handling for API requests and file operations.
# - Add rate limiting or delays if required by TCIA API usage policies.
# - Consider parallel downloads for efficiency.
# - Implement logic to handle potentially large dataset sizes and disk space.

import argparse
import os
import requests
import json
from tqdm import tqdm
import time

# --- Configuration (Placeholders - Update from TCIA Docs) ---
TCIA_API_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1" # Example base URL, verify!
# API_KEY = "YOUR_TCIA_API_KEY" # Obtain from TCI if needed

# --- Helper Functions ---

def make_tcia_request(endpoint, params=None, api_key=None):
    """ Helper function to make requests to the TCIA API. """
    headers = {}
    if api_key:
        headers['api_key'] = api_key # Verify header name from TCIA docs
    
    url = f"{TCIA_API_BASE_URL}/{endpoint}"
    print(f"Requesting: {url} with params: {params}")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # TCIA API might return JSON or other formats (like CSV for some queries)
        # Check content type or assume JSON for now
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            # Handle non-JSON responses if necessary (e.g., CSV parsing)
            print(f"Received non-JSON response (Content-Type: {response.headers.get('Content-Type')}). Returning raw text.")
            return response.text # Or handle differently based on expected format
            
    except requests.exceptions.RequestException as e:
        print(f"Error making TCIA API request to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response status: {e.response.status_code}")
             print(f"Response text: {e.response.text}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from {url}")
        print(f"Response text: {response.text}")
        return None

def download_image(series_instance_uid, output_dir, api_key=None):
    """ Downloads images for a given series instance UID. """
    # TODO: Verify the correct endpoint and parameters for downloading images/series
    endpoint = "getImage" # Placeholder endpoint name
    params = {'SeriesInstanceUID': series_instance_uid}
    
    headers = {}
    if api_key:
        headers['api_key'] = api_key
        
    url = f"{TCIA_API_BASE_URL}/{endpoint}"
    print(f"  Downloading series: {series_instance_uid}...")
    
    try:
        response = requests.get(url, headers=headers, params=params, stream=True)
        response.raise_for_status()

        # Determine filename (TCIA might provide it or use UID)
        # Example: using series UID, assuming zip format (verify actual format)
        filename = f"{series_instance_uid}.zip" 
        filepath = os.path.join(output_dir, filename)
        
        # Download with progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(filepath, 'wb') as f, tqdm(
            desc=f"  -> {filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
                
        print(f"  -> Saved to {filepath}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  Error downloading series {series_instance_uid}: {e}")
        return False
    except Exception as e:
         print(f"  Error saving file for series {series_instance_uid}: {e}")
         return False


# --- Main Download Logic ---

def main(args):
    print(f"--- TCIA Data Download Script (Placeholder) ---")
    print(f"Collection Name: {args.collection}")
    print(f"Output Directory: {args.output_dir}")
    
    api_key = args.api_key or os.environ.get("TCIA_API_KEY") # Get key from arg or env var
    # if not api_key:
    #     print("Warning: TCIA API Key not provided. Some requests might fail.")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Get Patients in Collection
    # TODO: Verify endpoint and parameters for getting patients
    print(f"\nFetching patients for collection '{args.collection}'...")
    patient_endpoint = "getPatient" # Placeholder
    patient_params = {'Collection': args.collection, 'format': 'json'} # Example params
    patients_data = make_tcia_request(patient_endpoint, params=patient_params, api_key=api_key)
    
    if not patients_data or not isinstance(patients_data, list):
        print("Error fetching patients or invalid format received. Exiting.")
        return
        
    print(f"Found {len(patients_data)} patients.")

    # 2. Iterate through Patients and Get Series
    total_series_to_download = 0
    series_list = []
    print("\nFetching series information for each patient...")
    for patient in tqdm(patients_data, desc="Fetching Series"):
        patient_id = patient.get('PatientID') # Assuming 'PatientID' is the key, verify!
        if not patient_id: 
            print("Warning: Skipping patient with missing ID.")
            continue
            
        # TODO: Verify endpoint and parameters for getting series for a patient
        series_endpoint = "getSeries" # Placeholder
        series_params = {'Collection': args.collection, 'PatientID': patient_id, 'format': 'json'} # Example
        series_data = make_tcia_request(series_endpoint, params=series_params, api_key=api_key)
        
        if series_data and isinstance(series_data, list):
            for series in series_data:
                series_uid = series.get('SeriesInstanceUID') # Assuming this key, verify!
                modality = series.get('Modality') # Assuming this key, verify!
                
                # Filter for CT modality if needed
                if series_uid and (not args.modality or modality == args.modality.upper()):
                    series_list.append(series_uid)
                    total_series_to_download += 1
        else:
            print(f"Warning: Could not fetch series for patient {patient_id}")
        
        # Optional: Add delay to avoid hitting API rate limits
        # time.sleep(0.1) 

    print(f"\nFound {total_series_to_download} series matching criteria.")

    # 3. Download Series
    if not series_list:
        print("No series found to download. Exiting.")
        return
        
    print("\nStarting series download...")
    download_count = 0
    download_errors = 0
    
    # Create a subdirectory for the collection
    collection_output_dir = os.path.join(args.output_dir, args.collection.replace(" ", "_"))
    os.makedirs(collection_output_dir, exist_ok=True)
    
    for series_uid in tqdm(series_list, desc="Downloading Series"):
        if download_image(series_uid, collection_output_dir, api_key):
            download_count += 1
        else:
            download_errors += 1
        # Optional: Add delay
        # time.sleep(0.1)

    print("\n--- Download Finished ---")
    print(f"Successfully downloaded: {download_count} series.")
    print(f"Failed downloads: {download_errors} series.")
    print(f"Data saved to: {collection_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download image series from TCIA using REST API.")
    parser.add_argument("--collection", type=str, required=True, help="Name of the TCIA collection (e.g., LIDC-IDRI).")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save downloaded series.")
    parser.add_argument("--modality", type=str, default="CT", help="Filter by modality (e.g., CT, MR). Default: CT.")
    parser.add_argument("--api_key", type=str, default=None, help="TCIA API Key (optional, can also use TCIA_API_KEY env var).")
    
    args = parser.parse_args()
    main(args)
