"""
test_data_collection.py
Runs and validates both YouTube data collection scripts:
1. scrape_youtube.py  (web-scraped video data)
2. api_youtube.py     (YouTube Data API)
"""

import subprocess
import os
import pandas as pd
from datetime import datetime

def run_script(script_name):
    print(f"\n Running {script_name} ...")
    result = subprocess.run(
        ["python", f"src/{script_name}"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(" STDERR:", result.stderr)
    print(" Done.\n")

def check_output(file_path):
    if not os.path.exists(file_path):
        print(f" Missing output file: {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f" {len(df):,} records found in {file_path}")
    print(f" Last modified: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
    print(f" Columns: {', '.join(df.columns)}\n")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Run both data collection scripts
    run_script("api_youtube.py")
    run_script("scrape_youtube.py")
    

    # Validate the outputs
    check_output("data/youtube_scraped_raw.csv")
    check_output("data/youtube_api_raw.csv")

    print(" Data collection test completed.")
