# iEEG Preprocessing and Frequency Band Analysis

## Short Description

This project uses Python to load, preprocess, and analyze intracranial EEG/iEEG data from an EDF file.

The goal is to build a beginner-friendly neuroscience analysis pipeline using `MNE-Python`, `pandas`, and `matplotlib`.

## What This Project Shows

- loading an EDF neural recording with `mne`
- inspecting basic recording metadata
- saving raw iEEG signal plots
- applying a 1-150 Hz band-pass filter
- applying 60 Hz and 120 Hz notch filters
- loading BIDS-style channel metadata from a `.tsv` file
- removing channels marked as bad
- computing power spectral density
- calculating average power in neural frequency bands
- saving plots and analysis results for a reproducible workflow

## Frequency Bands

| Band | Frequency Range |
| --- | --- |
| Theta | 4-8 Hz |
| Alpha | 8-13 Hz |
| Beta | 13-30 Hz |
| Gamma | 30-80 Hz |
| High Gamma | 80-150 Hz |

## Outputs

The script creates:

- `raw_ieeg_plot.png`
- `clean_ieeg_plot.png`
- `clean_power_spectrum.png`
- `band_power.png`
- `band_power_results.csv`

## Full Code

```python
import mne
import pandas as pd
import matplotlib.pyplot as plt


# Load the raw iEEG/ECoG signal from the EDF file
# preload=True means the signal data is loaded into memory so we can process it
raw = mne.io.read_raw_edf(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/sub-01_ses-01_task-game_run-01_ieeg.edf",
    preload=True
)

# Print basic information about the recording such as sampling rate, number of channels, filter settings, and duration
print(raw.info)


# Plot the raw unfiltered signal
# duration=10 shows 10 seconds of data
# n_channels=20 shows 20 channels at a time
# show=False means the plot will be saved instead of opened in a window
fig = raw.plot(duration=10, n_channels=20, show=False)

# Save the raw signal plot as an image file
fig.savefig(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/raw_ieeg_plot.png"
)
print("Saved raw_ieeg_plot.png")

# This keeps the original raw data unchanged by making a copy
raw_filtered = raw.copy()

# Apply a band-pass filter from 1 Hz to 150 Hz removing very slow drift below 1 Hz and high-frequency noise above 150 Hz
raw_filtered.filter(1, 150)

# Apply a notch filter at 60 Hz and 120 Hz dropping power of waves with frequency 60 or 120Hz to near zero values to filter out common electrical powerline noise
raw_filtered.notch_filter(freqs=[60, 120])

channels = pd.read_csv(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/sub-01_ses-01_task-game_run-01_channels.tsv",
    sep="\t"
)

# Find channels marked as bad in the metadata file and convert their names to the EDF style of 'POL (Channel number)-AV'
# This is due to naming differences in the dataset
# 3rd line says to only keep bad channel names that exist in the EDF file
bad_channels = channels.loc[channels["status"] == "bad", "name"].tolist()
bad_channels = [f"POL {ch}-AV" for ch in bad_channels]
bad_channels = [ch for ch in bad_channels if ch in raw_filtered.ch_names]

# Mark those channels as bad in MNE
raw_filtered.info["bads"] = bad_channels

# Create a cleaned copy with bad channels removed
raw_clean = raw_filtered.copy().drop_channels(bad_channels)

# Define frequency bands commonly used in EEG/iEEG analysis
frequency_bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
    "high_gamma": (80, 150)
}

# Compute power spectral density for frequencies 1-150 Hz for the cleaned data
psd = raw_clean.compute_psd(fmin=1, fmax=150)

# Get the actual PSD values and frequency values
psd_data = psd.get_data()
freqs = psd.freqs

# Calculate average power in each frequency band
band_powers = {}

for band_name, (fmin, fmax) in frequency_bands.items():
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    band_power = psd_data[:, freq_mask].mean()
    band_powers[band_name] = band_power

print("Average power by frequency band:")
print(band_powers)

# Convert the band power dictionary into a table and save as a CSV file
band_power_table = pd.DataFrame({
    "frequency_band": list(band_powers.keys()),
    "average_power": list(band_powers.values())
})
band_power_table.to_csv(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/band_power_results.csv",
    index=False
)

print("Saved band_power_results.csv")

# Bar plot of the average power for each frequency band
plt.figure(figsize=(8, 5))
plt.bar(band_powers.keys(), band_powers.values())
plt.xlabel("Frequency band")
plt.ylabel("Average power")
plt.title("Average iEEG Power by Frequency Band")
plt.tight_layout()
plt.savefig(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/band_power.png")
print("Saved band_power.png")

fig = raw_clean.plot(duration=10, n_channels=20, show=False)
fig.savefig(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/clean_ieeg_plot.png")
print("Saved clean_ieeg_plot.png")

# Compute the power spectral density up to 150 Hz, also called the PSD which shows how much signal power exists at different frequencies
fig = raw_clean.compute_psd(fmax=150).plot(show=False)
fig.savefig(
    "/Users/sinaenayati/Desktop/EEG data #2/sub-01/ses-01/ieeg/clean_power_spectrum.png")
print("Saved clean_power_spectrum.png")
```

## Notes For Future Improvement

- Use relative paths so the project runs on other computers.
- Save figures into a dedicated `figures/` folder.
- Save tables into a dedicated `results/` folder.
- Add a `requirements.txt` file with the Python packages.
- Consider adding event information later if the task timing files are available.
- Do not upload private or restricted raw patient data to a public GitHub repository unless the dataset license allows it.
