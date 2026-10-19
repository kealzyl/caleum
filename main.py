
#!/usr/bin/env python3
"""
Caleum Weather AI
A CUDA-enabled experimental weather prediction model.

Install:
    python -m pip install torch pandas numpy requests

Run:
    python main.py

PRETRAIN_DAYS controls how many days of historical weather data
are downloaded. It does NOT control how long training runs.

Features:
    - CUDA acceleration with CPU fallback
    - Configurable historical weather-data window
    - GRU neural network
    - Train for a configurable number of epochs
    - Best-model and latest checkpoints
    - Fine-tuning and resume support
    - 24-hour predictions
    - CSV forecast export
"""

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Caleum"

MODEL_FILE = "caleum_model.pt"
LAST_CHECKPOINT = "caleum_last.pt"
FORECAST_FILE = "caleum_forecast.csv"

# Location: Iași, Romania
LATITUDE = 47.1585
LONGITUDE = 27.6014
LOCAL_TIMEZONE = "Europe/Bucharest"

# Amount of historical weather data used for training.
# 30 days is approximately 720 hourly observations.
PRETRAIN_DAYS = 30

# Training computation settings.
DEFAULT_EPOCHS = 100
FINETUNE_EPOCHS = 50
BATCH_SIZE = 32

HIDDEN_SIZE = 128
INPUT_HOURS = 72
PREDICTION_HOURS = 24

LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 0.00001

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# NEURAL NETWORK
# ============================================================

class Caleum(nn.Module):
    """GRU network predicting the next 24 hours of weather."""

    def __init__(self, feature_count):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_count,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )

        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_SIZE,
                PREDICTION_HOURS * feature_count,
            ),
        )

        self.feature_count = feature_count

    def forward(self, x):
        sequence, _ = self.gru(x)
        last_state = sequence[:, -1, :]

        prediction = self.head(last_state)

        return prediction.reshape(
            -1,
            PREDICTION_HOURS,
            self.feature_count,
        )


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# CHECKPOINTS
# ============================================================

def load_checkpoint(path):
    """Load a Caleum checkpoint."""

    try:
        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )

    except Exception as safe_error:
        # Legacy compatibility: use only for your own trusted
        # checkpoint files, because unrestricted loading can
        # execute code embedded in an untrusted file.
        print("Trying legacy checkpoint compatibility...")

        try:
            return torch.load(
                path,
                map_location=DEVICE,
                weights_only=False,
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not load checkpoint {path}.\n"
                f"Safe-load error: {safe_error}\n"
                f"Legacy-load error: {error}"
            ) from error


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)

    return np.asarray(value, dtype=np.float32)


def save_checkpoint(
    path,
    model,
    optimizer,
    mean,
    std,
    best_val_loss,
    epoch,
):
    """Write a checkpoint atomically."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mean": torch.as_tensor(mean, dtype=torch.float32),
        "std": torch.as_tensor(std, dtype=torch.float32),
        "features": FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "input_hours": INPUT_HOURS,
        "prediction_hours": PREDICTION_HOURS,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": LOCAL_TIMEZONE,
        "best_val_loss": float(best_val_loss),
        "epoch": int(epoch),
    }

    temporary_path = path + ".tmp"

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def validate_checkpoint(checkpoint):
    if list(checkpoint.get("features", FEATURES)) != FEATURES:
        raise RuntimeError(
            "The checkpoint uses a different feature configuration."
        )

    if int(checkpoint.get("hidden_size", HIDDEN_SIZE)) != HIDDEN_SIZE:
        raise RuntimeError(
            "The checkpoint uses a different hidden size."
        )


# ============================================================
# DOWNLOAD HISTORICAL WEATHER
# ============================================================

def download_historical_weather(days=PRETRAIN_DAYS):
    """
    Download the requested number of calendar days.

    Open-Meteo's archive may have a delay, so this ends seven
    days before today's UTC date to avoid incomplete recent data.
    """

    if days < 3:
        raise ValueError("Use at least 3 days of weather data.")

    end_date = (
        datetime.now(timezone.utc).date()
        - timedelta(days=7)
    )

    start_date = end_date - timedelta(days=days - 1)

    print("\nDownloading historical weather...")
    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print(f"Start date: {start_date}")
    print(f"End date:   {end_date}")
    print(f"Requested data window: {days} days")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(FEATURES),
        "timezone": "UTC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=180,
    )
    response.raise_for_status()

    result = response.json()

    if "hourly" not in result:
        raise RuntimeError(
            f"Unexpected weather API response: {result}"
        )

    df = pd.DataFrame(result["hourly"])

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURES)
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )

    if len(df) < INPUT_HOURS + PREDICTION_HOURS + 24:
        raise RuntimeError(
            "Not enough usable hourly observations. "
            "Try a larger PRETRAIN_DAYS value."
        )

    print(f"Usable hourly observations: {len(df):,}")
    print(f"Actual data span: {df['time'].min()} to {df['time'].max()}")

    return df


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def make_sequences(values, start, stop, stride=1):
    """Build input windows and future targets."""

    xs = []
    ys = []

    last_start = stop - INPUT_HOURS - PREDICTION_HOURS

    for i in range(start, last_start + 1, stride):
        xs.append(values[i:i + INPUT_HOURS])

        ys.append(
            values[
                i + INPUT_HOURS:
                i + INPUT_HOURS + PREDICTION_HOURS
            ]
        )

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def prepare_data(df, mean=None, std=None):
    values = df[FEATURES].to_numpy(dtype=np.float32)

    split = int(len(values) * 0.80)

    # Compute normalization statistics from training data only.
    if mean is None or std is None:
        mean = values[:split].mean(axis=0)
        std = values[:split].std(axis=0)

        std[std < 1e-6] = 1.0

    else:
        mean = to_numpy(mean)
        std = to_numpy(std)
        std[std < 1e-6] = 1.0

    normalized = (values - mean) / std

    # Training targets remain inside the training interval.
    x_train, y_train = make_sequences(
        normalized,
        0,
        split,
    )

    # Validation windows may use the previous 72 hours as context,
    # but their future targets are in the validation interval.
    x_val, y_val = make_sequences(
        normalized,
        max(0, split - INPUT_HOURS),
        len(normalized),
    )

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError(
            "Not enough data to build training and validation sequences."
        )

    print(f"Training sequences: {len(x_train):,}")
    print(f"Validation sequences: {len(x_val):,}")

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        mean,
        std,
    )


# ============================================================
# TRAIN / FINE-TUNE / RESUME
# ============================================================

def train(mode="new", epochs=DEFAULT_EPOCHS, data_days=PRETRAIN_DAYS):
    """
    mode:
        new       - train a new model
        finetune  - fine-tune the best saved model
        resume    - continue from the latest checkpoint

    data_days controls the historical weather-data window.
    epochs controls the number of training passes over the dataset.
    """

    if epochs < 1:
        raise ValueError("Epoch count must be at least 1.")

    if data_days < 3:
        raise ValueError("Use at least 3 days of historical data.")

    if mode not in {"new", "finetune", "resume"}:
        raise ValueError(f"Unknown training mode: {mode}")

    checkpoint = None

    if mode == "finetune":
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"{MODEL_FILE} does not exist. Train a new model first."
            )

        checkpoint = load_checkpoint(MODEL_FILE)
        validate_checkpoint(checkpoint)

    elif mode == "resume":
        if not os.path.exists(LAST_CHECKPOINT):
            raise FileNotFoundError(
                f"{LAST_CHECKPOINT} does not exist."
            )

        checkpoint = load_checkpoint(LAST_CHECKPOINT)
        validate_checkpoint(checkpoint)

    df = download_historical_weather(data_days)

    if checkpoint is not None:
        mean = to_numpy(checkpoint["mean"])
        std = to_numpy(checkpoint["std"])

        (
            x_train,
            y_train,
            x_val,
            y_val,
            mean,
            std,
        ) = prepare_data(df, mean, std)

    else:
        (
            x_train,
            y_train,
            x_val,
            y_val,
            mean,
            std,
        ) = prepare_data(df)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_x = torch.from_numpy(x_val).to(DEVICE)
    val_y = torch.from_numpy(y_val).to(DEVICE)

    model = Caleum(len(FEATURES)).to(DEVICE)

    if checkpoint is not None:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    learning_rate = (
        FINETUNE_LEARNING_RATE
        if mode == "finetune"
        else LEARNING_RATE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    if mode == "resume" and checkpoint is not None:
        optimizer_state = checkpoint.get(
            "optimizer_state_dict"
        )

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    start_epoch = 0

    if mode == "resume" and checkpoint is not None:
        start_epoch = int(checkpoint.get("epoch", 0))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    loss_fn = nn.HuberLoss()

    if checkpoint is not None:
        best_val_loss = float(
            checkpoint.get("best_val_loss", float("inf"))
        )
    else:
        best_val_loss = float("inf")

    print("\n" + "=" * 60)
    print("CALEUM WEATHER AI")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    print(f"Parameters: {parameter_count(model):,}")
    print(f"Historical data window: {data_days} days")
    print(f"Epochs this session: {epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    session_start = time.monotonic()
    completed_epochs = 0

    try:
        for session_epoch in range(1, epochs + 1):
            model.train()

            total_loss = 0.0
            batches = 0

            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                prediction = model(xb)
                loss = loss_fn(prediction, yb)

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                total_loss += loss.item()
                batches += 1

            model.eval()

            with torch.no_grad():
                validation_prediction = model(val_x)

                val_loss = loss_fn(
                    validation_prediction,
                    val_y,
                ).item()

            train_loss = total_loss / max(batches, 1)
            scheduler.step(val_loss)

            completed_epochs += 1
            absolute_epoch = start_epoch + session_epoch

            elapsed = (
                time.monotonic() - session_start
            ) / 60

            print(
                f"Epoch {absolute_epoch:05d}"
                f" | Train loss: {train_loss:.5f}"
                f" | Validation loss: {val_loss:.5f}"
                f" | Elapsed: {elapsed:.1f} min"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    MODEL_FILE,
                    model,
                    optimizer,
                    mean,
                    std,
                    best_val_loss,
                    absolute_epoch,
                )

                print("  Saved new best model.")

            # Save the latest completed epoch so training can resume.
            save_checkpoint(
                LAST_CHECKPOINT,
                model,
                optimizer,
                mean,
                std,
                best_val_loss,
                absolute_epoch,
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

        # Last completed epoch is already saved.
        print(f"Latest checkpoint: {LAST_CHECKPOINT}")

    elapsed = (
        time.monotonic() - session_start
    ) / 60

    print("\nTraining session complete.")
    print(f"Epochs completed this session: {completed_epochs}")
    print(f"Elapsed: {elapsed:.1f} minutes")
    print(f"Best validation loss recorded: {best_val_loss:.5f}")
    print(f"Best model: {MODEL_FILE}")
    print(f"Latest checkpoint: {LAST_CHECKPOINT}")


# ============================================================
# 24-HOUR FORECAST
# ============================================================

def predict():
    if not os.path.exists(MODEL_FILE):
        print("No trained model found. Train Caleum first.")
        return

    checkpoint = load_checkpoint(MODEL_FILE)
    validate_checkpoint(checkpoint)

    model = Caleum(len(FEATURES)).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nDownloading recent weather observations...")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(FEATURES),
        "past_days": 4,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()

    if "hourly" not in result:
        raise RuntimeError(
            f"Unexpected forecast API response: {result}"
        )

    recent = pd.DataFrame(result["hourly"])

    recent["time"] = pd.to_datetime(
        recent["time"],
        utc=True,
    )

    for feature in FEATURES:
        recent[feature] = pd.to_numeric(
            recent[feature],
            errors="coerce",
        )

    now = pd.Timestamp.now(tz="UTC").floor("h")

    # Keep observations up to the current hour; don't accidentally
    # use future forecast values as if they were observations.
    recent = recent[
        recent["time"] <= now
    ].dropna(subset=FEATURES)

    recent = (
        recent.sort_values("time")
        .drop_duplicates(subset=["time"])
        .tail(INPUT_HOURS)
    )

    if len(recent) < INPUT_HOURS:
        raise RuntimeError(
            "Not enough recent hourly values for prediction."
        )

    mean = to_numpy(checkpoint["mean"])
    std = to_numpy(checkpoint["std"])
    std[std < 1e-6] = 1.0

    values = recent[FEATURES].to_numpy(
        dtype=np.float32
    )

    normalized = (values - mean) / std

    x = torch.from_numpy(
        normalized
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(x).cpu().numpy()[0]

    prediction = prediction * std + mean

    humidity_index = FEATURES.index(
        "relative_humidity_2m"
    )
    pressure_index = FEATURES.index(
        "surface_pressure"
    )
    wind_index = FEATURES.index(
        "wind_speed_10m"
    )
    rain_index = FEATURES.index(
        "precipitation"
    )

    prediction[:, humidity_index] = np.clip(
        prediction[:, humidity_index],
        0,
        100,
    )

    prediction[:, pressure_index] = np.maximum(
        prediction[:, pressure_index],
        0,
    )

    prediction[:, wind_index] = np.maximum(
        prediction[:, wind_index],
        0,
    )

    prediction[:, rain_index] = np.maximum(
        prediction[:, rain_index],
        0,
    )

    first_time = (
        recent["time"].iloc[-1]
        + pd.Timedelta(hours=1)
    )

    times = pd.date_range(
        start=first_time,
        periods=PREDICTION_HOURS,
        freq="h",
    ).tz_convert(LOCAL_TIMEZONE)

    forecast = pd.DataFrame(
        prediction,
        columns=FEATURES,
    )

    forecast.insert(0, "time", times)

    print("\n" + "=" * 95)
    print("CALEUM — EXPERIMENTAL 24-HOUR WEATHER PREDICTION")
    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print("=" * 95)

    print(
        f"{'Time':<23}"
        f"{'Temp °C':>10}"
        f"{'Humidity %':>12}"
        f"{'Pressure hPa':>15}"
        f"{'Wind km/h':>12}"
        f"{'Rain mm':>10}"
    )

    for _, row in forecast.iterrows():
        print(
            f"{row['time'].strftime('%Y-%m-%d %H:%M'):<23}"
            f"{row['temperature_2m']:>10.1f}"
            f"{row['relative_humidity_2m']:>12.1f}"
            f"{row['surface_pressure']:>15.1f}"
            f"{row['wind_speed_10m']:>12.1f}"
            f"{row['precipitation']:>10.2f}"
        )

    forecast.to_csv(
        FORECAST_FILE,
        index=False,
    )

    print(f"\nForecast saved to {FORECAST_FILE}")
    print(
        "Experimental predictions only; compare them against "
        "official forecasts and actual observations."
    )


# ============================================================
# MENU
# ============================================================

def main():
    print("\n" + "=" * 52)
    print("                 CALEUM WEATHER AI")
    print("=" * 52)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is unavailable. Running on CPU.")

    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print(f"Default historical window: {PRETRAIN_DAYS} days")

    while True:
        print("\n" + "-" * 52)
        print("1. Pretrain a new model")
        print("2. Predict the next 24 hours")
        print("3. Fine-tune the best model")
        print("4. Resume the latest checkpoint")
        print("5. Exit")

        choice = input("\nSelect an option: ").strip()

        try:
            if choice == "1":
                days_text = input(
                    f"Historical weather-data days [{PRETRAIN_DAYS}]: "
                ).strip()

                days = int(
                    days_text or PRETRAIN_DAYS
                )

                epochs_text = input(
                    f"Training epochs [{DEFAULT_EPOCHS}]: "
                ).strip()

                epochs = int(
                    epochs_text or DEFAULT_EPOCHS
                )

                train(
                    mode="new",
                    epochs=epochs,
                    data_days=days,
                )

            elif choice == "2":
                predict()

            elif choice == "3":
                days_text = input(
                    f"Historical weather-data days [{PRETRAIN_DAYS}]: "
                ).strip()

                days = int(
                    days_text or PRETRAIN_DAYS
                )

                epochs_text = input(
                    f"Fine-tuning epochs [{FINETUNE_EPOCHS}]: "
                ).strip()

                epochs = int(
                    epochs_text or FINETUNE_EPOCHS
                )

                train(
                    mode="finetune",
                    epochs=epochs,
                    data_days=days,
                )

            elif choice == "4":
                days_text = input(
                    f"Historical weather-data days [{PRETRAIN_DAYS}]: "
                ).strip()

                days = int(
                    days_text or PRETRAIN_DAYS
                )

                epochs_text = input(
                    f"Additional epochs [{DEFAULT_EPOCHS}]: "
                ).strip()

                epochs = int(
                    epochs_text or DEFAULT_EPOCHS
                )

                train(
                    mode="resume",
                    epochs=epochs,
                    data_days=days,
                )

            elif choice == "5":
                print("Caleum shutting down.")
                break

            else:
                print("Invalid option.")

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
            FileNotFoundError,
            KeyError,
        ) as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
