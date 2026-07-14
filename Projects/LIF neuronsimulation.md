# Single Neuron Simulation

## Short Description

This project simulates a simple **leaky integrate-and-fire neuron** in Python.

The neuron starts at a resting membrane potential. At each time step, input current pushes the voltage upward while leak pulls it back toward rest. If the voltage reaches a threshold, the neuron spikes and resets.

This project helped me practice Python loops, lists, functions, plotting with `matplotlib`, and connecting code to a computational neuroscience concept.

## What The Code Shows

- how membrane voltage changes over time
- how leak affects voltage
- how a spike threshold works
- how to store voltage values in a list
- how to plot a voltage trace
- how to mark spike times on a graph

## Full Code

```python
import matplotlib.pyplot as plt


V_rest = -70
V = V_rest
threshold = -55
input_current = 3
dt = 1

V_trace = []
spike_times = []
time_points = []

for step in range(10):
    time = step * dt
    time_points.append(time)

    leak = 0.1 * (V_rest - V)
    V = V + leak + input_current
    V_trace.append(round(V, 2))

    if V >= threshold:
        print("Spike!")
        spike_times.append(time)
        V = V_rest
    else:
        print("No spike")

print(time_points)
print(V_trace)
print(spike_times)

plt.plot(time_points, V_trace)
plt.axhline(threshold, color="red")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.scatter(spike_times, [threshold] * len(spike_times), color="black")
plt.title("Leaky Integrate-and-Fire Neuron")
plt.show()
```

## How To Run It

Install `matplotlib`:

```bash
pip install matplotlib
```

Run the file:

```bash
python lif_neuron.py
```

If that does not work, try:

```bash
python3 lif_neuron.py
```

## Notes

This is a first version of the model. A future version could add:

- more time steps
- different input currents
- a refractory period
- multiple neurons
- a cleaner function-based structure
