1.	Ensure python and git are downloaded onto your local machine
2.	Clone the repository onto your local machine
3.	Connect to the repository through where you locally installed it
4.	Setup a python virtual environment if not set up already
5.	Install dependencies using pip install -r requirements.txt.
6.	Create a .env file in the project root with your own YouTube API key:
 	api_key=your_api_key_here
7.	Run the entire pipeline using:
 	python src/run_all.py
 	This executes data scraping, API collection, preprocessing, feature engineering, model training, and visualization automatically. Note: The web scraping process in Step 1 will take approximately 12 minutes to complete due to the large keyword set and YouTube page load times.
8.	All processed data and output visualizations will be saved in the data/ directory.
