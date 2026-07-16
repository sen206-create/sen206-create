<div align="center">

# Hi, I'm Sina Enayati

### Computational Neuroscience and Scientific Programming

I am learning how to use code for neural data analysis, constructing computational models, and becoming a stronger scientific programmer.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)


</div>

---

## About Me

I am interested in computational neuroscience and I am developing my programming skills through below listed projects.

My goal is to write code that helps me understand:

- how neurons and neural systems process information
- how data from the brain can be analyzed and visualized
- how mathematical models can explain behavior and cognition
- how to make scientific code cleaner, clearer, and more reproducible

---

## What I Am Learning

| Skill                      | Why it matters                                                                   |
| -------------------------- | -------------------------------------------------------------------------------- |
| Python                     | Building computational neuroscience models and analysing neural data             |
| Portfolio design           | Curating a public portfolio to display my progress                               |
| Computational neuroscience | Learning neural coding, spike trains, Bayesian inference, and EEG signal analysis|

---

## Recent Projects (Descending Chronological Order)

### iEEG Analysis Journey: From First Failure to Generalization

A Python project documenting my path from iEEG preprocessing to tap-vs-not-tap machine learning, written as a two-part learning story.

**Part 1:** I started with one iEEG recording, cleaned the signal, created tap and not-tap examples, and trained a logistic regression model on raw voltage windows. That first model failed at about 49% accuracy, which showed me that raw `channels x time` data was not a good first representation.

I then rebuilt the model using band-power features from theta, alpha, beta, gamma, and high-gamma activity. That version improved to about 78.1% accuracy on the same recording.

**Part 2:** The next question is harder: does the band-power model still work on recordings it has never seen before? I am expanding the analysis across other sessions and subjects to test generalization, not just performance on one file.

**Python skills:** MNE-Python, pandas, NumPy, scikit-learn, feature extraction, classification  
**Neuroscience idea:** iEEG preprocessing, finger-tap events, frequency-band power, model failure, and generalization across recordings

The [project file](projects/ieeg-preprocessing-frequency-analysis.md) includes the analysis story, failed baseline, improved feature model, Part 2 generalization plan, and code.

<p align="center">· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·</p>

### University Of Washington, Computational Neuroscicence Online Course

A collection of Python models and exercises from the University of Washington online computational neuroscience course. This project includes membrane dynamics, integrate-and-fire neurons, Poisson variability, population coding, noisy spike timing, synaptic input, and eigenvector-based learning.

**Python skills:** NumPy, matplotlib, simulation loops, plotting, loading data with pickle
**Neuroscience ideas:** neural coding, membrane voltage, spike timing, synaptic input, and learning rules

The [project file](Projects/uw-neuro-course-models.md) includes short descriptions of the models and the code for each one.

<p align="center">· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·</p>

### Leaky-integrate-and-fire Neuron Model

A simple Python simulation of a leaky integrate-and-fire neuron.

This project shows how a neuron's membrane potential changes over time, how input current affects voltage, and how a spike happens when the voltage reaches a threshold.

**Python skills:** loops, lists, variables, plotting with `matplotlib`  
**Neuroscience idea:** membrane potential, leak, threshold, and spike reset

The [Single Neuron Simulation](Projects/LIF-Neuron-Simulation.md) project file includes a short description and the full code.

---

## Current Goals

- Get better at Python for scientific computing
- Learn how to write cleaner and more organized code
- Learn how to analyse EEG data
- Practice using GitHub to document my learning
- Familarise with machine learning concepts

---

## Connect

- GitHub: `https://github.com/YOUR_GITHUB_USERNAME`
- LinkedIn: `www.linkedin.com/in/sina-enayati-25b878328`
- Email: `senayati2006@gmail.com`

---

<div align="center">

### Learning Python by building tools for computational neuroscience.

</div>
