# pse-stocks

## Project Overview

This repository contains the source code used for my undergraduate thesis.

## Installation

### Prerequisites
Ensure you have Python installed on your system. TensorFlow 2 supports:
- Python: 3.8–3.11
- Operating Systems:
    - Ubuntu 16.04 or later
    - Windows 7 or later (with C++ redistributable)
    - macOS 10.12.6 (Sierra) or later (no GPU support)
    - WSL2 via Windows 10 19044+ (Experimental GPU support)

Verify Python installation:
```sh
python3 --version # python --version for Windows
```
If Python is not installed, download it from the [official website](https://www.python.org/).


### Installing Python Dependencies

With the virtual environment activated, install the required packages:

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### Installing GPU Dependencies

Make sure to download the GPU drivers if not running on the cloud [here](https://www.nvidia.com/en-us/drivers/)

## Usage

### Data Scraping

To scrape PSE stock data, run:

```sh
python scrape_data.py
```


This script utilizes `tickers.txt` to determine which stocks to fetch and stores the data in the `data/` directory.

### Jupyter Notebook

The provided notebook runs the methodology outlined in the paper. It skips the data collection part since it is already collected via `scrape_data.py`, and ends with the backtesting simulations.

Run it via:

```sh
jupyter notebook notebook.ipynb
```

## Project Directory Structure

The repository organizes data as follows:

```
/pse-stocks
    ├── data/
    └── predictions/
    └── results/
```

- `data/`: Contains scraped stock data from MarketWatch.
- `predictions/`: Contains actual vs predicted price plots for the final model.
- `results/`: Contains backtesting results for each strategies.

## Model

The project saves the final trained model to a file called `model.keras`, which can be loaded for other uses.