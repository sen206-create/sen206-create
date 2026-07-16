# Single Neuron Simulation

## Short Description

This project simulates a simple **leaky integrate-and-fire neuron** in Python.

The neuron starts at a resting membrane potential. At each time step, input current pushes the voltage upward while leak pulls it back toward rest. If the voltage reaches a threshold, the neuron spikes and resets.

This project helped me practice Python functions, loops, lists, plotting with `matplotlib`, and connecting code to a computational neuroscience concept.

## What The Code Shows

- how membrane voltage changes over time
- how leak pulls voltage back toward rest
- how a spike threshold works
- how spike times can be stored
- how to save a voltage-trace figure

## Main Code

```python
from pathlib import Path

import matplotlib.pyplot as plt


def simulate_lif_neuron(
    v_rest=-70,
    threshold=-55,
    input_current=3,
    dt=1,
    steps=10,
):
    """Simulate a simple leaky integrate-and-fire neuron."""
    voltage = v_rest
    voltage_trace = []
    spike_times = []
    time_points = []

    for step in range(steps):
        time = step * dt
        time_points.append(time)

        # Leak pulls voltage back toward resting potential.
        leak = 0.1 * (v_rest - voltage)
        voltage = voltage + leak + input_current
        voltage_trace.append(round(voltage, 2))

        if voltage >= threshold:
            print(f"Time {time} ms: Spike!")
            spike_times.append(time)
            voltage = v_rest
        else:
            print(f"Time {time} ms: No spike")

    return time_points, voltage_trace, spike_times


def plot_voltage_trace(time_points, voltage_trace, spike_times, threshold):
    """Plot membrane potential over time and mark spike times."""
    project_dir = Path(__file__).resolve().parents[1]
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(time_points, voltage_trace, marker="o", label="Membrane potential")
    plt.axhline(threshold, color="red", linestyle="--", label="Spike threshold")
    plt.scatter(
        spike_times,
        [threshold] * len(spike_times),
        color="black",
        zorder=3,
        label="Spike",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Leaky Integrate-and-Fire Neuron")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lif_neuron_plot.png", dpi=200)
    plt.close()
```
## Example Output

The script prints whether the neuron spikes at each time step and saves this plot:

[Leaky integrate-and-fire neuron plot](Projects/Figures/lif_neuron_plot.png)

## Notes

This is a first version of the model. A future version could add:

- more time steps
- different input currents
- a refractory period
- multiple neurons
- a comparison between different parameter values
