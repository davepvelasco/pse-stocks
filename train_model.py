import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# GPU optimization settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Hyperparameters
HYPERPARAMS = {
    "num_units_min": 50,
    "num_units_max": 200,
    "num_layers_min": 1,
    "num_layers_max": 3,
    "dropout_min": 0.0,
    "dropout_max": 0.5,
    "learning_rate_min": 1e-4,
    "learning_rate_max": 1e-2,
    "batch_sizes": [16, 32, 64],
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    "n_trials": 10,  # Number of Optuna trials
    "epochs": 50,  # Number of epochs per trial
}

# Stock-related parameters
STOCKS = [
    "AC",
    "ACEN",
    "AEV",
    "AGI",
    "ALI",
    "BDO",
    "BLOOM",
    "BPI",
    "CNPF",
    "CNVRG",
    "DMC",
    "EMI",
    "GLO",
    "GTCAP",
    "ICT",
    "JFC",
    "JGS",
    "LTG",
    "MBT",
    "MER",
    "MONDE",
    "NIKL",
    "PGOLD",
    "SCC",
    "SM",
    "SMC",
    "SMPH",
    "TEL",
    "URC",
    "WLCON",
]
sequence_length = 60
features = [
    "Open",
    "Close",
    "SMA_fast",
    "SMA_slow",
    "RSI",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "spread",
]
target = "target"

# Define technical indicator parameters
SMA_FAST_PERIOD = 10
SMA_SLOW_PERIOD = 50
RSI_PERIOD = 14
BOLLINGER_BANDS_PERIOD = 20
BOLLINGER_BANDS_STD_DEV = 2.0


# Helper function to load stock data
def load_stock_data(ticker, data_dir="data"):
    file_path = Path(data_dir) / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file for {ticker} not found at {file_path}.")

    # Load the CSV file
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Remove commas and convert columns to numeric
    for col in df.columns:
        df[col] = df[col].replace(",", "", regex=True).astype(float)

    return df


def add_technical_indicators(df):
    # Add SMA indicators
    df["SMA_fast"] = ta.sma(df["Close"], length=SMA_FAST_PERIOD)
    df["SMA_slow"] = ta.sma(df["Close"], length=SMA_SLOW_PERIOD)

    # Add RSI
    df["RSI"] = ta.rsi(df["Close"], length=RSI_PERIOD)

    # Add Bollinger Bands
    bbands = ta.bbands(
        df["Close"], length=BOLLINGER_BANDS_PERIOD, std=BOLLINGER_BANDS_STD_DEV
    )
    df["BB_upper"] = bbands[f"BBU_{BOLLINGER_BANDS_PERIOD}_{BOLLINGER_BANDS_STD_DEV}"]
    df["BB_middle"] = bbands[f"BBM_{BOLLINGER_BANDS_PERIOD}_{BOLLINGER_BANDS_STD_DEV}"]
    df["BB_lower"] = bbands[f"BBL_{BOLLINGER_BANDS_PERIOD}_{BOLLINGER_BANDS_STD_DEV}"]

    # Add spread feature and remove "High" and "Low" columns
    df["spread"] = df["High"] - df["Low"]
    df.drop(columns=["High", "Low"], inplace=True)

    # Fill NaN values (from technical indicators)
    df.bfill(inplace=True)
    return df


# Helper function to prepare data for a single stock
def prepare_stock_data(stock, sequence_length, features, target):
    # Load and preprocess data
    df = load_stock_data(stock)
    df = add_technical_indicators(df)

    # Ensure proper date range
    start_date = df.index.min()
    end_date = df.index.max()
    if not (start_date.year == 2004 and end_date.year == 2024):
        print(
            f"Excluding {stock} due to date range: {start_date.date()} to {end_date.date()}"
        )
        return None

    # Save the processed dataframe as a CSV
    df.to_csv(f"data/{stock}_processed.csv")

    # Add target column
    df["target"] = df["Close"].shift(-1)  # Predict next close price
    df.dropna(inplace=True)

    # Split data
    train_size = int(0.6 * len(df))
    val_size = int(0.2 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size : train_size + val_size]
    test_df = df[train_size + val_size :]

    # Print date ranges
    print(f"Stock: {stock}")
    print(f"  Train set: {train_df.index[0]} to {train_df.index[-1]}")
    print(f"  Validation set: {val_df.index[0]} to {val_df.index[-1]}")
    print(f"  Test set: {test_df.index[0]} to {test_df.index[-1]}")

    # Scale the data
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df), columns=val_df.columns, index=val_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df), columns=test_df.columns, index=test_df.index
    )

    # Create rolling windows
    def create_rolling_windows(data, seq_len, feature_cols, target_col):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data.iloc[i : i + seq_len][feature_cols].values)
            y.append(data.iloc[i + seq_len][target_col])
        return np.array(X), np.array(y)

    X_train, y_train = create_rolling_windows(
        train_scaled, sequence_length, features, target
    )
    X_val, y_val = create_rolling_windows(val_scaled, sequence_length, features, target)
    X_test, y_test = create_rolling_windows(
        test_scaled, sequence_length, features, target
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, df, test_df


# Prepare data for all stocks
stock_data = {}
for stock in STOCKS:
    result = prepare_stock_data(stock, sequence_length, features, target)
    if result is not None:  # Only include stocks that pass the criteria
        stock_data[stock] = result


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,  # Number of epochs to wait for improvement
    restore_best_weights=True,
)


def build_model(input_shape, num_units, num_layers, dropout):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for _ in range(num_layers):
        model.add(
            LSTM(
                num_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
            )
        )
        if dropout > 0.0:
            model.add(Dropout(dropout))
    model.add(
        LSTM(
            num_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
        )
    )
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def objective(trial):
    # Hyperparameters to tune
    num_units = trial.suggest_int(
        "num_units", HYPERPARAMS["num_units_min"], HYPERPARAMS["num_units_max"]
    )
    num_layers = trial.suggest_int(
        "num_layers", HYPERPARAMS["num_layers_min"], HYPERPARAMS["num_layers_max"]
    )
    dropout = trial.suggest_float(
        "dropout", HYPERPARAMS["dropout_min"], HYPERPARAMS["dropout_max"]
    )
    learning_rate = trial.suggest_float(
        "learning_rate",
        HYPERPARAMS["learning_rate_min"],
        HYPERPARAMS["learning_rate_max"],
        log=True,
    )
    batch_size = trial.suggest_categorical("batch_size", HYPERPARAMS["batch_sizes"])

    # Build the LSTM model
    model = build_model(
        input_shape=(sequence_length, len(features)),
        num_units=num_units,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )

    # Training
    for stock, data in stock_data.items():
        X_train, y_train, X_val, y_val, _, _, _, _, _ = data
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0,
            epochs=EXPERIMENT_CONFIG["epochs"],
            callbacks=[early_stopping],
        )

    # Evaluate on validation data
    val_loss = 0
    for stock, data in stock_data.items():
        X_val, y_val = data[2], data[3]
        val_loss += model.evaluate(X_val, y_val, verbose=0)
    return val_loss / len(stock_data)


# Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=EXPERIMENT_CONFIG["n_trials"])

# Best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train final model with best hyperparameters
best_params = study.best_params
final_model = build_model(
    input_shape=(sequence_length, len(features)),
    num_units=best_params["num_units"],
    num_layers=best_params["num_layers"],
    dropout=best_params["dropout"],
)
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
    loss="mse",
)
for stock, data in stock_data.items():
    X_train, y_train, X_val, y_val, _, _, _, _, _ = data
    final_model.fit(
        X_train,
        y_train,
        batch_size=best_params["batch_size"],
        validation_data=(X_val, y_val),
        epochs=10,
        callbacks=[early_stopping],
        verbose=1,
    )

# Save the final model
checkpoint_path = "model.keras"
final_model.save(checkpoint_path)
plot_model(final_model, to_file="model.png")
print(f"Best model saved to {checkpoint_path}")
