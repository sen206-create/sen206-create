import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


DATA_ROOT = Path(os.environ.get("IEEG_DATA_ROOT",
                 Path.home() / "Desktop" / "EEG data #2"))
OUTPUT_PATH = DATA_ROOT / "full_band_power_ml_dataset.csv"

TMIN = -0.5
TMAX = 1.0
RANDOM_SEED = 42
TAP_TRIGGER = "101"

FREQUENCY_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
    "high_gamma": (80, 150),
}


np.random.seed(RANDOM_SEED)


def get_matching_channels_path(edf_path):
    base_name = edf_path.name.replace("_ieeg.edf", "")
    return edf_path.with_name(base_name + "_channels.tsv")


def clean_raw(edf_path, channels_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    raw_filtered = raw.copy()
    raw_filtered.filter(1, 150, verbose=False)
    raw_filtered.notch_filter(freqs=[60, 120], verbose=False)

    channels = pd.read_csv(channels_path, sep="\t")
    bad_channels = channels.loc[channels["status"] == "bad", "name"].tolist()
    bad_channels = [f"POL {ch}-AV" for ch in bad_channels]
    bad_channels = [ch for ch in bad_channels if ch in raw_filtered.ch_names]

    return raw_filtered.copy().drop_channels(bad_channels)


def extract_band_power_features(window, sfreq, frequency_bands):
    """Return one band-power feature row for one channels x time window."""
    freqs, psd = welch(window, fs=sfreq, axis=1)
    features = {}

    for band_name, (fmin, fmax) in frequency_bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        features[band_name] = psd[:, freq_mask].mean()

    return features


def make_rows_for_file(edf_path):
    channels_path = get_matching_channels_path(edf_path)

    if not channels_path.exists():
        print(f"Skipping {edf_path.name}: missing channels file")
        return []

    raw_clean = clean_raw(edf_path, channels_path)
    events, event_id = mne.events_from_annotations(raw_clean, verbose=False)

    if len(events) == 0:
        print(f"Skipping {edf_path.name}: no annotations/events found")
        return []

    tap_event_id = {
        event_name: event_code
        for event_name, event_code in event_id.items()
        if TAP_TRIGGER in str(event_name)
    }

    if not tap_event_id:
        print(f"Skipping {edf_path.name}: no tap trigger {TAP_TRIGGER} found")
        return []

    epochs = mne.Epochs(
        raw_clean,
        events,
        event_id=tap_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X_tap = epochs.get_data()

    if len(X_tap) == 0:
        print(f"Skipping {edf_path.name}: no usable tap epochs")
        return []

    tap_event_codes = list(tap_event_id.values())
    tap_samples = events[np.isin(events[:, 2], tap_event_codes), 0]
    raw_data = raw_clean.get_data()
    window_samples = X_tap.shape[2]
    num_not_tap = len(X_tap)
    sfreq = raw_clean.info["sfreq"]

    rows = []

    for window in X_tap:
        row = extract_band_power_features(window, sfreq, FREQUENCY_BANDS)
        row.update(
            {
                "label": 1,
                "label_name": "tap",
                "file": edf_path.name,
                "subject": edf_path.parts[-4],
                "session": edf_path.parts[-3],
            }
        )
        rows.append(row)

    not_tap_count = 0
    attempts = 0
    max_attempts = num_not_tap * 100

    while not_tap_count < num_not_tap and attempts < max_attempts:
        attempts += 1
        start = np.random.randint(0, raw_data.shape[1] - window_samples)
        end = start + window_samples

        too_close_to_tap = np.any(
            (tap_samples >= start) & (tap_samples <= end))

        if too_close_to_tap:
            continue

        window = raw_data[:, start:end]
        row = extract_band_power_features(window, sfreq, FREQUENCY_BANDS)
        row.update(
            {
                "label": 0,
                "label_name": "not_tap",
                "file": edf_path.name,
                "subject": edf_path.parts[-4],
                "session": edf_path.parts[-3],
            }
        )
        rows.append(row)
        not_tap_count += 1

    print(f"Processed {edf_path.name}: {len(rows)} examples")
    return rows


edf_files = sorted(DATA_ROOT.glob("sub-*/ses-*/ieeg/*_ieeg.edf"))

all_rows = []

for edf_path in edf_files:
    all_rows.extend(make_rows_for_file(edf_path))

if not all_rows:
    raise RuntimeError("No usable EDF files were processed.")

dataset = pd.DataFrame(all_rows)
dataset.to_csv(OUTPUT_PATH, index=False)

print("Saved full band-power dataset:", OUTPUT_PATH)
print("Dataset shape:", dataset.shape)
print(dataset["label_name"].value_counts())
