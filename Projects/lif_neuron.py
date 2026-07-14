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
    plt.figure(figsize=(8, 5))
    plt.plot(time_points, voltage_trace,
             marker="o", label="Membrane potential")
    plt.axhline(threshold, color="red", linestyle="--",
                label="Spike threshold")
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
    plt.show()


def main():
    threshold = -55
    time_points, voltage_trace, spike_times = simulate_lif_neuron(
        threshold=threshold)

    print("\nTime points:")
    print(time_points)

    print("\nVoltage trace:")
    print(voltage_trace)

    print("\nSpike times:")
    print(spike_times)

    plot_voltage_trace(time_points, voltage_trace, spike_times, threshold)


if __name__ == "__main__":
    main()
