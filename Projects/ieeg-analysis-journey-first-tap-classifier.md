# iEEG Analysis Journey: Building the First Tap Classifier

## Short Description

This project is Part 1 of my iEEG analysis journey.

The goal was to learn how to take a real intracranial EEG recording and turn it into a machine learning dataset. I started with one recording, cleaned the signal, found finger-tapping events, created tap and not-tap examples, and trained a first logistic regression classifier.

The classifier did not work well. That failure became the point of the project.

## Dataset

This project uses the OpenNeuro dataset `ds005931`, **Visuomotor_task**.

Dataset DOI: `https://doi.org/10.18112/openneuro.ds005931.v1.0.0`  
License: `CC0`

Raw data are not included in this repository. The code expects the dataset to be downloaded separately.

## Project Question

```text
Can I classify finger-tap activity from iEEG voltage windows?
```

The first version of the project uses:

- one iEEG recording
- event annotations for finger tapping
- tap epochs around each event
- random not-tap windows from the same recording
- logistic regression as a first simple classifier

## Stage 1: Preprocess the iEEG Signal

Before doing machine learning, I needed to load and clean the signal.

### What This Step Does

- loads the raw EDF file
- filters the signal between 1 and 150 Hz
- removes 60 Hz and 120 Hz line noise
- reads the channel metadata
- removes channels marked as bad
- saves plots so I can inspect the recording

```python
from pathlib import Path

import mne
import pandas as pd


data_root = Path.home() / "Desktop" / "EEG data #2"
recording_dir = data_root / "sub-01" / "ses-01" / "ieeg"

edf_path = recording_dir / "sub-01_ses-01_task-game_run-01_ieeg.edf"
channels_path = recording_dir / "sub-01_ses-01_task-game_run-01_channels.tsv"

raw = mne.io.read_raw_edf(edf_path, preload=True)
raw_filtered = raw.copy()
raw_filtered.filter(1, 150)
raw_filtered.notch_filter(freqs=[60, 120])

channels = pd.read_csv(channels_path, sep="\t")
bad_channels = channels.loc[channels["status"] == "bad", "name"].tolist()
bad_channels = [f"POL {ch}-AV" for ch in bad_channels]
bad_channels = [ch for ch in bad_channels if ch in raw_filtered.ch_names]

raw_clean = raw_filtered.copy().drop_channels(bad_channels)

fig = raw_clean.plot(duration=10, n_channels=20, show=False)
fig.savefig("clean_ieeg_plot.png")
```

![Clean_power_spectrum](Figures/clean_power_spectrum.png)

## Stage 2: Create Tap and Not-Tap Examples

The dataset contains annotations for the task. In this dataset, trigger `101` marks finger tapping. MNE reads this as `Trigger-101`.

The machine learning labels are:

```text
1 = tap
0 = not tap
```

```python
import numpy as np
import mne


events, event_id = mne.events_from_annotations(raw_clean)

print("Event IDs:")
print(event_id)

print("First 10 events:")
print(events[:10])

tap_event_id = {
    event_name: event_code
    for event_name, event_code in event_id.items()
    if "101" in str(event_name)
}

epochs = mne.Epochs(
    raw_clean,
    events,
    event_id=tap_event_id,
    tmin=-0.5,
    tmax=1.0,
    baseline=None,
    preload=True,
)

X_tap = epochs.get_data()
y_tap = np.ones(len(X_tap))

window_samples = X_tap.shape[2]

raw_data = raw_clean.get_data()
tap_event_codes = list(tap_event_id.values())
tap_samples = events[np.isin(events[:, 2], tap_event_codes), 0]

not_tap_examples = []

while len(not_tap_examples) < len(X_tap):
    start = np.random.randint(0, raw_data.shape[1] - window_samples)
    end = start + window_samples

    too_close_to_tap = np.any((tap_samples >= start) & (tap_samples <= end))

    if not too_close_to_tap:
        not_tap_examples.append(raw_data[:, start:end])

X_not_tap = np.array(not_tap_examples)
y_not_tap = np.zeros(len(X_not_tap))

X_all = np.concatenate([X_tap, X_not_tap], axis=0)
y_all = np.concatenate([y_tap, y_not_tap])

print("X shape:", X_all.shape)
print("y shape:", y_all.shape)
```

## First Classifier: Raw Voltage Windows

For the first model, I gave logistic regression the raw voltage values directly.

Each example started as:

```text
channels x time_points
```

The model needs a flat table, so I reshaped each example into one long row.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_flat = X_all.reshape(X_all.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(
    X_flat,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all,
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
```

## Result

The raw-window logistic regression model performed around chance level:

```text
Accuracy: ~49%
```

This means the model was not reliably separating tap from not-tap windows.

## What I Learned

The failure was useful because it showed me that raw voltage windows are a difficult first representation for a simple model.

The model was given thousands of time-point values, but it did not know which patterns mattered. It also had no built-in understanding of neural frequency bands.

The next step was to ask a better question:

```text
Instead of giving the model every voltage value, can I summarize each window using meaningful iEEG features?
```

That question led to Part 2:

[iEEG Analysis Journey: Band Power and Generalization](ieeg-analysis-journey-band-power-generalization.md)
