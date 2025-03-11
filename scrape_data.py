import os
import random
import time
from pathlib import Path

import pandas as pd
from seleniumbase import SB

# Load tickers from a file
with open("tickers.txt", "r") as file:
    TICKERS = [line.strip() for line in file if line.strip()]


# Generate the download link for a given year range and ticker
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

# Path to save data
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)


def download_stock_data(ticker):
    output_filename = output_dir / f"{ticker}.csv"

    # Skip if already downloaded
    if output_filename.exists():
        print(f"Data for {ticker} already exists. Skipping download.")
        return

    print(f"Processing ticker: {ticker}")
    all_data = pd.DataFrame()

    try:
        with SB(uc=True, headed=True) as sb:
            sb.activate_cdp_mode("about:blank")
            sb.cdp.open("https://www.marketwatch.com/investing/stock/ac/download-data")
            print("If there's a CAPTCHA, solve it manually.")
            sb.sleep(10)

            download_dir = Path("downloaded_files")

            for year in range(start_year, end_year):
                link = generate_download_link(ticker, year, year + 1)
                print(f"Downloading {ticker} stock data for year {year}")
                sb.cdp.open(link)
                sb.sleep(5)  # Wait for the download

                # Find the latest downloaded file
                files = sorted(
                    download_dir.glob("*.csv"), key=os.path.getmtime, reverse=True
                )
                if files:
                    latest_file = files[0]
                    data = pd.read_csv(latest_file)
                    all_data = pd.concat([all_data, data], ignore_index=True)
                    os.remove(latest_file)
                else:
                    raise FileNotFoundError(f"No CSV file found for {ticker} in {year}")

                time.sleep(random.uniform(2, 5))  # Human delay

        # Ensure data integrity
        if all_data.empty:
            raise ValueError(f"No data downloaded for ticker: {ticker}")

        all_data["Date"] = pd.to_datetime(all_data["Date"], format="%m/%d/%Y")
        all_data.sort_values(by="Date", inplace=True)
        all_data.to_csv(output_filename, index=False)
        print(f"Data for {ticker} saved to {output_filename}")

    except Exception as e:
        print(f"Critical error while processing {ticker}: {e}")
        exit(1)  # Terminate program immediately


# Iterate over each ticker
target_tickers = TICKERS[:5]
for ticker in target_tickers:
    download_stock_data(ticker)
