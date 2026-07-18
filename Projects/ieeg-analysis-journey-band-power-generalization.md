# iEEG Analysis Journey: Band Power and Generalization

## Short Description

This project is Part 2 of my iEEG analysis journey.

In Part 1, my first logistic regression model failed when I used raw channels x time voltage windows. In this part, I redesigned the model using frequency-band power features, then tested whether the improvement generalized to other recordings.

The result was more realistic than perfect: the band-power model improved performance on one recording, but it did not generalize well across unseen recordings.

## Dataset

This project uses the OpenNeuro dataset `ds005931`, **Visuomotor_task**.

Dataset DOI: `https://doi.org/10.18112/openneuro.ds005931.v1.0.0`  
License: `CC0`

Raw data are not included in this repository. The code expects the dataset to be downloaded separately.

## Project Question
### (1a) Can frequency-band power features improve tap-vs-not-tap classification,
### (1b) Do those features generalize to new iEEG recordings?

## Stage 3: Band-Power Features

Instead of feeding the model every voltage value, I summarized each window using average power in standard frequency bands.

This stage continues from Part 1, where `X_all` contains tap and not-tap windows and `y_all` contains their labels.

| Band | Frequency Range |
| --- | --- |
| Theta | 4-8 Hz |
| Alpha | 8-13 Hz |
| Beta | 13-30 Hz |
| Gamma | 30-80 Hz |
| High gamma | 80-150 Hz |

This makes the model easier to interpret because each feature has a neuroscience meaning.

```python
import numpy as np
from scipy.signal import welch


frequency_bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
    "high_gamma": (80, 150),
}


def extract_band_power_features(X, sfreq, frequency_bands):
    features = []

    for trial in X:
        trial_features = []

        freqs, psd = welch(trial, fs=sfreq, axis=1)

        for band_name, (fmin, fmax) in frequency_bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = psd[:, freq_mask].mean()
            trial_features.append(band_power)

        features.append(trial_features)

    return np.array(features)
```

## Band-Power Logistic Regression

The new model used only five features per example:

```text
theta, alpha, beta, gamma, high_gamma
```

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


X_features = extract_band_power_features(X_all, sfreq=1000, frequency_bands=frequency_bands)

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all,
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred, target_names=["not_tap", "tap"], zero_division=0))
```

## Single-Recording Result

On the same recording used in Part 1, the band-power model improved substantially.

| Model | Feature Representation | Accuracy |
| --- | --- | --- |
| First logistic regression | Raw `channels x time` voltage windows | ~49% |
| Improved logistic regression | Frequency-band power | ~78.1% |

The confusion matrix was:
![Band-power classifier confusion matrix](Figures/confusion_matrix_band_power.png)


## Stage 4: Testing Generalization

The single-recording result was encouraging, but it raised a harder question:

`Does the model work on recordings it has never seen before?`

To test that, I expanded the dataset across other sessions and subjects.

The full dataset builder creates one table with this structure:

```text
subject | session | file | theta | alpha | beta | gamma | high_gamma | label
```

The full scripts are:

```python
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
```

- [05_train_full_band_power_classifier.py](../scripts/05_train_full_band_power_classifier.py)

## Generalization Result

When I tested the band-power model across more recordings, performance dropped back near chance.

| Evaluation | Accuracy | Interpretation |
| --- | --- | --- |
| Random split across all examples | ~48.0% | The model did not learn a reliable rule across the larger dataset. |
| Held-out recording split | ~49.4% | The model did not generalize to recordings it had never seen before. |

The held-out recording confusion matrix was:

```text
[[295 378]
 [303 370]]
```

This means the model was almost equally confused between tap and not-tap examples.

## What I Learned

Part 2 taught me that improving accuracy on one recording is not the same as building a model that generalizes.

The band-power features were useful inside one recording, but the model did not hold up across new recordings. That suggests the classifier may have learned recording-specific patterns rather than a general tap-vs-not-tap signature.

This is still a strong result for learning because it shows the difference between:

```text
single-recording performance
```

and:

```text
generalization to unseen neural recordings
```

## Next Steps

- Improve normalization across recordings.
- Save Part 2 classification reports and confusion matrices as result files.
- Compare band-power features with other feature types.
- Try session-aware and subject-aware validation.
- Only move to neural networks or transformers after the feature-based baseline is understood.
