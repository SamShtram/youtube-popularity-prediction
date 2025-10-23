"""
YouTube Popularity Prediction - Full Pipeline
---------------------------------------------

This script runs the complete workflow:

1. Scrape trending YouTube data (web scraping)
2. Collect trending YouTube data via API
3. Preprocess and clean both datasets
4. Engineer features
5. Train and evaluate models
6. Generate visualizations

Before running:
- Ensure you have Python 3.10+ and required libraries installed (see requirements.txt)
- Ensure the `.env` file exists at the project root with:
      YOUTUBE_API_KEY=your_api_key_here
- The 'data' directory will be created automatically if not present.
"""

import os
import subprocess
import sys

def run_step(description, command):
    print(f"\n=== {description} ===")
    result = subprocess.run([sys.executable, command], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:\n", result.stderr)

def main():
    print("Starting full YouTube Popularity Prediction pipeline...\n")

    # Step 1: Scrape data (web scraping)
    if os.path.exists("src/scrape_youtube.py"):
        run_step("Step 1: Scraping YouTube Data", "src/scrape_youtube.py")
    else:
        print("Skipping scraping (src/scrape_youtube.py not found).")

    # Step 2: Collect data via API
    print("\nReminder: Ensure your YouTube API key is set in .env before running this step.")
    if os.path.exists("src/api_youtube.py"):
        run_step("Step 2: Collecting YouTube API Data", "src/api_youtube.py")
    else:
        print("Skipping API collection (src/api_youtube.py not found).")

    # Step 3: Preprocess and clean
    run_step("Step 3: Preprocessing Raw Data", "src/preprocessing.py")

    # Step 4: Feature Engineering
    run_step("Step 4: Feature Engineering", "src/feature_engineering.py")

    # Step 5: Train and evaluate models (Scraped and API)
    run_step("Step 5A: Model Training (Scraped Data)", "src/model_scraped.py")
    run_step("Step 5B: Model Training (API Data)", "src/model_api.py")

    # Step 6: Visualization
    run_step("Step 6: Generating Visualizations", "src/visualization.py")

    print("\nPipeline execution complete. All processed data and visualizations are in the /data folder.")
    print("Final outputs:")
    print("  - youtube_scraped_clean.csv")
    print("  - youtube_api_clean.csv")
    print("  - youtube_scraped_features.csv")
    print("  - youtube_api_features.csv")
    print("  - youtube_scraped_ready.csv")
    print("  - youtube_api_ready.csv")
    print("  - model_comparison.png, feature_importance.png, views_vs_duration.png")

if __name__ == "__main__":
    main()
