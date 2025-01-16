import json
import random
import time
from io import StringIO
from pathlib import Path

import curl_cffi.requests as requests
import pandas as pd

# Load headers from a JSON file
with open("headers.json", "r") as file:
    HEADERS = json.load(file)

# Load tickers from a file
with open("tickers.txt", "r") as file:
    TICKERS = [line.strip() for line in file if line.strip()]

# Function to generate the download link for a given year range and ticker
def generate_download_link(ticker, start_year, end_year):
    base_url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}/downloaddatapartial"
    startdate = f"01/01/{start_year}%2000:00:00"
    enddate = f"01/01/{end_year}%2000:00:00"
    params = (
        f"startdate={startdate}&enddate={enddate}&daterange=d30&"
        f"frequency=p1d&csvdownload=true&downloadpartial=false&newdates=false&countrycode=ph"
    )
    return f"{base_url}?{params}"

# Generate links for the last 20 years
start_year = 2004
end_year = 2025

# Initialize a session for persistent headers and connection pooling
session = requests.Session()

# Update session headers
session.headers.update(HEADERS)

# Path to save data
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

# Iterate over each ticker and download data
for ticker in TICKERS:
    print(f"Processing ticker: {ticker}")
    all_data = pd.DataFrame()
    for year in range(start_year, end_year):
        link = generate_download_link(ticker, year, year + 1)
        try:
            response = session.get(link, allow_redirects=True)
            if response.status_code == 200:
                data = pd.read_csv(StringIO(response.text))
                all_data = pd.concat([all_data, data], ignore_index=True)
            else:
                print(f"Failed to download data for {ticker} link: {link}. HTTP Status Code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading data for {ticker} link: {link}. Error: {e}")

        # Add a realistic delay between requests
        time.sleep(random.uniform(2, 5))

    # Ensure the Date column is in datetime format
    if not all_data.empty:
        all_data['Date'] = pd.to_datetime(all_data['Date'], format='%m/%d/%Y')

        # Sort the data chronologically
        all_data.sort_values(by="Date", inplace=True)

        # Save the data to a CSV file in the output directory
        output_filename = output_dir / f"{ticker}.csv"
        all_data.to_csv(output_filename, index=False)
        print(f"Data for {ticker} saved to {output_filename}")
    else:
        print(f"No data downloaded for ticker: {ticker}")
