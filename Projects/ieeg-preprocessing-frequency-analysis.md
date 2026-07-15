# iEEG Analysis Journey: From Preprocessing to Machine Learning

## Short Description

This project documents my first full iEEG analysis workflow, starting with preprocessing intracranial EEG data and moving toward machine learning classification.

The main goal was to learn how to turn real neural recordings into a machine learning dataset. I started with raw voltage windows, saw that a simple classifier failed, and then redesigned the model around neuroscience-informed band-power features.

## Dataset

This project uses the OpenNeuro dataset `ds005931`, **Visuomotor_task**.

Dataset DOI: `https://doi.org/10.18112/openneuro.ds005931.v1.0.0`  
License: `CC0`

Raw data are not included in this repository. The analysis code expects the dataset to be downloaded separately.

## Project Story

The project developed in three stages:

1. **Preprocess the iEEG signal**
2. **Build a first tap-vs-not-tap classifier using raw voltage windows**
3. **Improve the model by switching to interpretable band-power features**

The first classifier did not work well, which became an important part of the project. It showed that raw flattened time-series data was not a good first representation for logistic regression.

---

## Stage 1: iEEG Preprocessing

The first step was to load and clean the iEEG recording.

### What This Stage Does

- loads the raw EDF file with `MNE-Python`
- applies a 1-150 Hz band-pass filter
- applies 60 Hz and 120 Hz notch filters
- reads the channel metadata file
- removes channels marked as bad
- computes power spectral density
- calculates average power in standard frequency bands
- saves plots and CSV outputs

### Frequency Bands

| Band | Frequency Range |
| --- | --- |
| Theta | 4-8 Hz |
| Alpha | 8-13 Hz |
| Beta | 13-30 Hz |
| Gamma | 30-80 Hz |
| High Gamma | 80-150 Hz |

### Preprocessing Code

```python
import mne
import pandas as pd
import matplotlib.pyplot as plt


edf_path = "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/sub-01_ses-01_task-game_run-01_ieeg.edf"
channels_path = "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/sub-01_ses-01_task-game_run-01_channels.tsv"


# Load the raw iEEG/ECoG signal from the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Make a copy so the original raw data stays unchanged
raw_filtered = raw.copy()

# Apply a band-pass filter from 1 Hz to 150 Hz
raw_filtered.filter(1, 150)

# Remove 60 Hz and 120 Hz electrical line noise
raw_filtered.notch_filter(freqs=[60, 120])

# Load channel metadata
channels = pd.read_csv(channels_path, sep="\t")

# Find channels marked as bad in the metadata file
bad_channels = channels.loc[channels["status"] == "bad", "name"].tolist()

# Convert channel names from TSV style to EDF style
bad_channels = [f"POL {ch}-AV" for ch in bad_channels]

# Keep only bad channel names that exist in the EDF file
bad_channels = [ch for ch in bad_channels if ch in raw_filtered.ch_names]

# Create a cleaned copy with bad channels removed
raw_clean = raw_filtered.copy().drop_channels(bad_channels)

# Plot the cleaned signal
fig = raw_clean.plot(duration=10, n_channels=20, show=False)
fig.savefig("clean_ieeg_plot.png")

# Compute and save the power spectrum
fig = raw_clean.compute_psd(fmax=150).plot(show=False)
fig.savefig("clean_power_spectrum.png")
```

---

## Stage 2: First Machine Learning Attempt

After preprocessing, I used the dataset annotations to find finger-tapping events.

According to the dataset description, trigger code `101` marks finger tapping. MNE detected these annotations as `Trigger-101`.

### Goal

Train a classifier to distinguish:

```text
1 = finger tap
0 = not tap
```

### Creating Tap and Not-Tap Examples

```python
import numpy as np
import mne
import pandas as pd


edf_path = "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/sub-01_ses-01_task-game_run-01_ieeg.edf"
channels_path = "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/sub-01_ses-01_task-game_run-01_channels.tsv"
output_path = "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/tap_vs_not_tap_dataset.npz"


raw = mne.io.read_raw_edf(edf_path, preload=True)
events, event_id = mne.events_from_annotations(raw)

raw_filtered = raw.copy()
raw_filtered.filter(1, 150)
raw_filtered.notch_filter(freqs=[60, 120])

channels = pd.read_csv(channels_path, sep="\t")
bad_channels = channels.loc[channels["status"] == "bad", "name"].tolist()
bad_channels = [f"POL {ch}-AV" for ch in bad_channels]
bad_channels = [ch for ch in bad_channels if ch in raw_filtered.ch_names]

raw_clean = raw_filtered.copy().drop_channels(bad_channels)

# Create epochs around finger-tapping events
epochs = mne.Epochs(
    raw_clean,
    events,
    event_id=event_id,
    tmin=-0.5,
    tmax=1.0,
    baseline=None,
    preload=True
)

# Tap examples: trials x channels x time points
X_tap = epochs.get_data()
y_tap = np.ones(len(X_tap))

# Create not-tap examples from random windows
tap_samples = events[:, 0]
raw_data = raw_clean.get_data()
window_samples = X_tap.shape[2]
num_not_tap = len(X_tap)

not_tap_examples = []

while len(not_tap_examples) < num_not_tap:
    start = np.random.randint(0, raw_data.shape[1] - window_samples)
    end = start + window_samples

    too_close_to_tap = np.any((tap_samples >= start) & (tap_samples <= end))

    if not too_close_to_tap:
        window = raw_data[:, start:end]
        not_tap_examples.append(window)

X_not_tap = np.array(not_tap_examples)
y_not_tap = np.zeros(len(X_not_tap))

# Combine tap and not-tap examples
X_all = np.concatenate([X_tap, X_not_tap], axis=0)
y_all = np.concatenate([y_tap, y_not_tap])

np.savez(output_path, X=X_all, y=y_all)

print("Saved tap_vs_not_tap_dataset.npz")
print("X shape:", X_all.shape)
print("y shape:", y_all.shape)
```

### First Classifier: Raw Voltage Windows

The first classifier used the raw `channels x time` windows directly. Each window was flattened into one long feature vector.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = np.load(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/tap_vs_not_tap_dataset.npz"
)

X = data["X"]
y = data["y"]

# Flatten each trial from channels x time into one long vector
X_flat = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(
    X_flat,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
```

### What Happened

This raw-window classifier performed around chance level, with accuracy near `0.49`.

It mostly predicted the tap class and failed to reliably identify the not-tap examples.

This was useful because it showed that raw flattened voltage traces are not a good first feature representation for this problem. The model was given thousands of raw time-point features, but no structured understanding of neural frequency content.

---

## Stage 3: Improved Model With Band-Power Features

The next model uses features that are more meaningful for EEG/iEEG analysis.

Instead of feeding the classifier every voltage value, I summarize each window using average power in frequency bands.

### Why This Is Better

Raw iEEG windows can contain thousands of time-point features.

Band-power features are smaller and easier to interpret:

```text
theta power
alpha power
beta power
gamma power
high-gamma power
```

Each feature has a neuroscience meaning, which makes the model easier to understand.

### Band-Power Feature Classifier

```python
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


data = np.load(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/tap_vs_not_tap_dataset.npz"
)

X = data["X"]
y = data["y"]

# Sampling rate from the dataset
sfreq = 1000

frequency_bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
    "high_gamma": (80, 150)
}


def extract_band_power_features(X, sfreq, frequency_bands):
    """Convert iEEG windows into average band-power features."""
    features = []

    for trial in X:
        trial_features = []

        # trial shape: channels x time_points
        freqs, psd = welch(trial, fs=sfreq, axis=1)

        for band_name, (fmin, fmax) in frequency_bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = psd[:, freq_mask].mean()
            trial_features.append(band_power)

        features.append(trial_features)

    return np.array(features)


X_features = extract_band_power_features(X, sfreq, frequency_bands)

print("Band-power feature shape:", X_features.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# StandardScaler puts features on a similar scale before logistic regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Band-power classifier accuracy:", accuracy)
print(classification_report(y_test, y_pred, zero_division=0))

results = pd.DataFrame({
    "feature": list(frequency_bands.keys()),
    "description": [
        "4-8 Hz average power",
        "8-13 Hz average power",
        "13-30 Hz average power",
        "30-80 Hz average power",
        "80-150 Hz average power"
    ]
})

results.to_csv("band_power_feature_descriptions.csv", index=False)
```

### How To Report The Result

After running the band-power classifier, report the measured accuracy here:

```markdown
The band-power model achieved an accuracy of `ADD_RESULT_HERE`.
```

Do not claim it worked until the result is measured. If it improves over the raw-window model, the project story becomes:

```text
raw voltage features performed poorly
band-power features gave a better and more interpretable model
```

If it does not improve, that is still scientifically useful. It means the next step should be better labels, better non-tap sampling, subject/session splits, or a more expressive model.

---

## What I Learned

This project taught me that EEG/iEEG machine learning is not just about choosing a model.

The representation matters.

The first classifier failed because raw voltage windows were too large and unstructured for a simple model. Moving to band-power features made the machine learning task more interpretable and closer to standard EEG/iEEG analysis.

## Next Steps

- Run the band-power classifier and record the measured accuracy.
- Save the classification report to a text file.
- Plot confusion matrices for both classifiers.
- Extend the dataset from one recording to multiple sessions.
- Try a neural network only after the feature-based baseline is understood.
- Eventually test a transformer on structured iEEG windows.
