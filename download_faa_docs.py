import requests
from pathlib import Path
from tqdm import tqdm
import subprocess
import time
import hashlib

DATA_DIR = Path("data/far_aim")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Stable local names + latest FAA URLs
DOCS = {
    "AIM.pdf": "https://www.faa.gov/air_traffic/publications/media/AIM_Basic_w_Chg_1_and_2_dtd_1-22-26.pdf",
    "PHAK.pdf": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/faa-h-8083-25c.pdf",
    "Airplane_Flying_Handbook.pdf": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/airplane_handbook/00_afh_full.pdf",
    "Instrument_Procedures_Handbook.pdf": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/instrument_procedures_handbook/FAA-H-8083-16B.pdf",
    "Aviation_Weather_Handbook.pdf": "https://www.faa.gov/sites/faa.gov/files/FAA-H-8083-28B.pdf",
    "Risk_Management_Handbook.pdf": "https://www.faa.gov/sites/faa.gov/files/2022-06/risk_management_handbook_2A.pdf",
    "Aviation_Instructors_Handbook.pdf": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/aviation_instructors_handbook/aviation_instructors_handbook.pdf",
}

def get_file_hash(filepath):
    """Simple hash to detect meaningful changes"""
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read(1024*1024)).hexdigest()  # First 1MB hash is enough

def download_file(url, local_name):
    filepath = DATA_DIR / local_name
    old_hash = get_file_hash(filepath)
    
    print(f"📥 Checking {local_name}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, "wb") as f, tqdm(
            desc=local_name[:50],
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                size = f.write(data)
                bar.update(size)

        new_hash = get_file_hash(filepath)
        if old_hash != new_hash:
            print(f"✅ Updated: {local_name}")
            return True
        else:
            print(f"✅ Already up to date: {local_name}")
            return False

    except Exception as e:
        print(f"❌ Failed {local_name}: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Smart FAA Document Updater...\n")
    updated = False

    for local_name, url in DOCS.items():
        if download_file(url, local_name):
            updated = True
        time.sleep(2)

    print("\n🎉 Download complete!")

    if updated:
        print("🔄 Changes detected — Rebuilding Aviation Brain...")
        try:
            subprocess.run(["python", "ingest_faraim.py"], check=True)
            print("✅ Ingestion completed successfully!")
        except Exception as e:
            print(f"⚠️ Ingestion failed: {e}")
    else:
        print("✅ No updates found.")

    print(f"\nFiles are in: {DATA_DIR.resolve()}")
