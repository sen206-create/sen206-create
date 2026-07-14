# University of Washington Computational Neuroscience Course Models

## Short Description

This project collects Python models and exercises I built while working through the University of Washington online computational neuroscience course.

The folder contains small simulations and analyses that connect programming to core computational neuroscience ideas, including membrane dynamics, integrate-and-fire neurons, Poisson variability, population coding, synaptic input, and eigenvector-based learning.

## What This Project Shows

- using Python to simulate neuron voltage over time
- using `numpy` for numerical computation
- using `matplotlib` to visualize neural models
- decoding a stimulus direction using population vectors
- exploring how noise affects spike timing
- working with eigenvectors and input correlations

## Files In The Course Folder

| File | What it does |
| --- | --- |
| `poissoin4.py` | Checks Poisson-like variability by plotting firing-rate mean vs variance |
| `pop_coding4.py` | Uses population coding to estimate an angle from neural responses |
| `int and fire` | Simulates a basic integrate-and-fire neuron |
| `intfirenoise` | Simulates how input noise changes interspike interval distributions |
| `Test5` | Simulates passive membrane charging and estimates the membrane time constant |
| `test6` | Simulates synaptic input with alpha functions and spike-rate adaptation |
| `test7` | Implements a simple eigenvector/learning-rule style calculation |
| `Input correlation eigenvectors` | Computes eigenvectors from an input correlation matrix |
| `import pickletest4` | Loads the tuning dataset from a pickle file |

## Model 1: Poisson Variability Check

This script loads neural tuning data and compares the mean and variance of firing rates.

In a Poisson process, the mean and variance are often similar. Plotting them against each other is a useful way to check whether spike count variability looks Poisson-like.

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('tuning_3.4.pickle', 'rb') as f:
    data = pickle.load(f)

for neuron in ['neuron1', 'neuron2', 'neuron3', 'neuron4']:
    mean_fire = np.mean(data[neuron], axis=0)
    var_fire = np.var(data[neuron], axis=0)

    plt.scatter(mean_fire, var_fire, label=neuron)

plt.plot([0, max(mean_fire)], [0, max(mean_fire)], '--')
plt.xlabel('Mean firing rate')
plt.ylabel('Variance')
plt.title('Poisson check: variance vs mean')
plt.legend()
plt.show()
```

## Model 2: Population Vector Coding

This script estimates a stimulus direction from the activity of multiple neurons.

Each neuron contributes a preferred direction vector, weighted by its response relative to its maximum firing rate.

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('pop_coding_3.4.pickle', 'rb') as f:
    datap = pickle.load(f)

with open('tuning_3.4.pickle', 'rb') as f:
    data = pickle.load(f)

rmax1 = np.max(np.mean(data['neuron1'], axis=0))
rmax2 = np.max(np.mean(data['neuron2'], axis=0))
rmax3 = np.max(np.mean(data['neuron3'], axis=0))
rmax4 = np.max(np.mean(data['neuron4'], axis=0))
r1 = np.mean(datap['r1'])
r2 = np.mean(datap['r2'])
r3 = np.mean(datap['r3'])
r4 = np.mean(datap['r4'])

v = (r1 / rmax1) * datap['c1'] \
    + (r2 / rmax2) * datap['c2'] \
    + (r3 / rmax3) * datap['c3'] \
    + (r4 / rmax4) * datap['c4']

angle = np.degrees(np.arctan2(v[0], v[1]))
angle = angle % 360

print("population vector:", v)
print("angle:", round(angle))
```

## Model 3: Passive Membrane Time Constant

This script simulates how membrane voltage changes in response to injected current.

It also compares the theoretical membrane time constant with the time measured from the simulated voltage trace.

```python
import numpy as np
import matplotlib.pyplot as plt

# input current
I = 10  # nA

# capacitance and leak resistance

C = 0.1  # nF
R = 100  # M ohms
tau = R*C  # = 0.1*100 nF-Mohms = 100*100 pF Mohms = 10 ms
print('C = %.3f nF' % C)
print('R = %.3f M ohms' % R)
print('tau = %.3f ms' % tau)
print('(Theoretical)')

# membrane potential equation dV/dt = - V/RC + I/C

tstop = 150  # ms

V_inf = I*R  # peak V (in mV)
tau = 0  # experimental (ms)

h = 0.2  # ms (step size)

V = 0  # mV
V_trace = [V]  # mV

for t in np.arange(h, tstop, h):

    # Euler method: V(t+h) = V(t) + h*dV/dt
    V = V + h*(- (V/(R*C)) + (I/C))

    # Verify membrane time constant
    if (not tau and (V > 0.6321*V_inf)):
        tau = t
        print('tau = %.3f ms' % tau)
        print('(Experimental)')

    # Stop current injection
    if t >= 0.6*tstop:
        I = 0

    V_trace += [V]
    if t % 10 == 0:
        plt.plot(np.arange(0, t+h, h), V_trace, color='r')
        plt.xlim(0, tstop)
        plt.ylim(0, V_inf)
        plt.draw()


plt.show()
```

## Model 4: Basic Integrate-and-Fire Neuron

This script simulates a neuron that integrates input current, spikes when it reaches threshold, and then enters a refractory period.

```python
import numpy as np
import matplotlib.pyplot as plt


# input current
I = 10  # nA

# capacitance and leak resistance
C = 1  # nF
R = 40  # M ohms

# I & F implementation dV/dt = - V/RC + I/C
# Using h = 1 ms step size, Euler method

V = 0
tstop = 200
abs_ref = 5  # absolute refractory period
ref = 0  # absolute refractory period counter
V_trace = []  # voltage trace for plotting
V_th = 10  # spike threshold

for t in range(tstop):

    if not ref:
        V = V - (V/(R*C)) + (I/C)
    else:
        ref -= 1
        V = 0.2 * V_th  # reset voltage

    if V > V_th:
        V = 50  # emit spike
        ref = abs_ref  # set refractory counter

    V_trace += [V]


plt.plot(V_trace)
plt.show()
```

## Model 5: Noise and Interspike Intervals

This script adds noise to the input current and measures how noise changes the distribution of interspike intervals.

```python
import numpy as np
import matplotlib.pyplot as plt

# Neuron parameters
C = 1          # nF
R = 40         # M ohms
tstop = 10000  # ms
abs_ref = 5    # refractory period
V_th = 10      # spike threshold
baseline_I = 1  # nA

# Noise amplitudes: 0, 0.5, 1.0, ..., 5.0
noise_values = np.arange(0, 5.5, 0.5)

for noiseamp in noise_values:

    V = 0
    ref = 0
    spiketimes = []

    # Create noisy input current
    I = baseline_I + noiseamp * np.random.normal(0, 1, tstop)

    # Run neuron simulation
    for t in range(tstop):

        if ref == 0:
            V = V - V / (R * C) + I[t] / C
        else:
            ref -= 1
            V = 0.2 * V_th

        if V > V_th:
            V = 50
            ref = abs_ref
            spiketimes.append(t)

    # Calculate interspike intervals
    isi = np.diff(spiketimes)

    # Need several ISIs to estimate a distribution
    if len(isi) >= 2:

        # X-axis for the smooth curve
        x = np.linspace(isi.min(), isi.max(), 300)

        # Start density values at zero
        y = np.zeros_like(x)

        # Controls how smooth the curve is
        bandwidth = 2

        # Add one small Gaussian curve around each ISI
        for value in isi:
            y += np.exp(
                -0.5 * ((x - value) / bandwidth) ** 2
            )

        # Normalise into probability density
        y = y / (
            len(isi)
            * bandwidth
            * np.sqrt(2 * np.pi)
        )

        plt.plot(x, y, label=f"Noise = {noiseamp:.1f}")

    else:
        print(f"Noise {noiseamp:.1f}: not enough spikes")

plt.xlabel("Interspike interval (ms)")
plt.ylabel("Probability density")
plt.title("ISI distributions across noise amplitudes")
plt.legend()
plt.show()
```

## Model 6: Synaptic Input and Spike-Rate Adaptation

This script uses a random input spike train, alpha-function synaptic conductance, and spike-rate adaptation.

It explores how changing the synaptic time-to-peak affects the output spike count.

```python
import time
import numpy as np
from numpy import concatenate as cc
import matplotlib.pyplot as plt

np.random.seed(0)
# I & F implementation dV/dt = - V/RC + I/C
h = 1.  # step size, Euler method, = dt ms
t_max = 200  # ms, simulation time period
tstop = int(t_max/h)  # number of time steps
ref = 0  # refractory period counter

# Generate random input spikes
# Note: This is not entirely realistic - no refractory period
# Also: if you change step size h, input spike train changes too...
thr = 0.9  # threshold for random spikes
spike_train = np.random.rand(tstop) > thr

# alpha func synaptic conductance
t_a = 100  # Max duration of syn conductance
t_peak_values = np.arange(0.5, 10.5, 0.5)  # ms
g_peak = 0.05  # nS (peak synaptic conductance)
# const
t_vec = np.arange(0, t_a + h, h)
# alpha func

# alpha func plot

# capacitance and leak resistance
C = 0.5  # nF
R = 40  # M ohms
print('C = {}'.format(C))
print('R = {}'.format(R))

# conductance and associated parameters to simulate spike rate adaptation

G_inc = 1/h
tau_ad = 2

# Initialize basic parameters
E_leak = -60  # mV, equilibrium potential
E_syn = 0  # Excitatory synapse (why is this excitatory?)
g_syn = 0  # Current syn conductance
V_th = -40  # spike threshold mV
V_spike = 50  # spike value mV
ref_max = int(4/h)  # Starting value of ref period counter
t_list = np.array([], dtype=int)
V = E_leak
V_trace = [V]
t_trace = [0]

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.arange(0, t_max, h), spike_train)
axs[0].set_title('Input spike train')

spike_counts = []

for t_peak in t_peak_values:

    const = g_peak / (t_peak * np.exp(-1))
    alpha_func = const * t_vec * np.exp(-t_vec / t_peak)

    V = E_leak
    g_ad = 0
    g_syn = 0
    t_list = np.array([], dtype=int)
    spike_count = 0
    ref = 0

    V_trace = [V]
    t_trace = [0]

    for t in range(tstop):

        # Compute input
        if spike_train[t]:  # check for input spike
            t_list = cc([t_list, [1]])

        # Calculate synaptic current due to current and past input spikes
        g_syn = np.sum(alpha_func[t_list])
        I_syn = g_syn*(E_syn - V)

        # Update spike times
        if np.any(t_list):
            t_list = t_list + 1
            if t_list[0] == t_a:  # Reached max duration of syn conductance
                t_list = t_list[1:]

        # Compute membrane voltage
        # Euler method: V(t+h) = V(t) + h*dV/dt
        if not ref:
            V = V + h*(-((V-E_leak)*(1+R*g_ad)/(R*C)) + (I_syn/C))
            g_ad = g_ad + h*(-g_ad/tau_ad)  # spike rate adaptation
        else:
            ref -= 1
            V = V_th - 10  # reset voltage after spike
            g_ad = 0

        # Generate spike
        if (V > V_th) and not ref:
            V = V_spike
            spike_count += 1
            ref = ref_max
            g_ad = g_ad + G_inc

        V_trace += [V]
        t_trace += [t*h]

    spike_counts.append(spike_count)


plt.figure()

plt.plot(
    t_peak_values,
    spike_counts,
    marker="o"
)

axs[1].plot(t_trace, V_trace)
plt.draw()
axs[1].set_title('Output spike train')
plt.show()
print("Spike Count:", spike_counts)
```

## Model 7: Eigenvectors and Input Correlations

This script computes the eigenvalues and eigenvectors of a correlation-like matrix, then identifies the principal eigenvector.

```python
import numpy as np


Q = np.array([
    [0.2, 0.1],
    [0.1, 0.3]
])

eigenvalues, eigenvectors = np.linalg.eig(Q)

print(eigenvalues)
print(eigenvectors)

largest_index = np.argmax(eigenvalues)

principal_eigenvector = eigenvectors[:, largest_index]

print("Principal eigenvector:")
print(principal_eigenvector)

ratio = principal_eigenvector[1] / principal_eigenvector[0]

print("w2 / w1 =", ratio)
candidate = np.array([1.055, 1.7013])

candidate_ratio = candidate[1] / candidate[0]

print(candidate_ratio)
```

## Model 8: Centered Input Data and Weight Learning

This script loads a dataset, mean-centers it, and updates a weight vector using an activity-dependent learning rule.

It then compares the learned direction with the principal eigenvector of a matrix.

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('c10p1.pickle', 'rb') as f:
    data = pickle.load(f)

points = data["c10p1"]

mean_point = np.mean(points, axis=0)
u = points - mean_point

print("Mean after centering:", np.mean(u, axis=0))

plt.scatter(u[:, 0], u[:, 1])
plt.axhline(0)
plt.axvline(0)
plt.xlabel("Centred x")
plt.ylabel("Centred y")
plt.title("Mean-centred input data")
plt.show()

alpha = 1
eta = 1
dt = 0.01
num_iterations = 100_000

w = np.random.rand(2)

num_points = len(u)

for t in range(num_iterations):
    index = t % num_points
    current_u = u[index]

    v = current_u @ w

    delta_w = dt*eta * (
        v * current_u
    )

    w = w + delta_w

print("Final weight vector:", w)
print("Length of final vector:", np.linalg.norm(w))

Q = np.array([[0.2, 0.1],
              [0.1, 0.3]])

vals, vecs = np.linalg.eig(Q)
w = vecs[:, np.argmax(vals)]

print("w =", w)
print("Length of final weight vector:", np.linalg.norm(w))
```

## How To Run These Models

These scripts use:

```bash
pip install numpy matplotlib
```

Some scripts also require the `.pickle` files from the course folder to be in the same folder as the script.



