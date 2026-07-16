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

### iEEG Analysis Journey (Part Two): Band Power and Generalization

A follow-up iEEG project where I improved the feature representation, then tested whether the improvement held up on unseen recordings.

Instead of feeding the model raw voltage windows, I summarized each window using theta, alpha, beta, gamma, and high-gamma power. On one recording, this improved logistic regression accuracy to about 78.1%.

Then I expanded across other sessions and subjects. The model dropped back to about 48-49% accuracy on the larger/unseen-recording tests, showing that single-recording success did not automatically generalize.

**Python skills:** SciPy, NumPy, pandas, scikit-learn, feature extraction, model evaluation  
**Neuroscience idea:** frequency-band power, model interpretability, and generalization across neural recordings

The [linked project file](projects/ieeg-analysis-journey-band-power-generalization.md) includes the band-power model, the cross-recording test, and the lesson from the model failing to generalize.

<p align="center">· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·</p>

### iEEG Analysis Journey (Part One): Building the First Tap Classifier

A Python project where I learned how to turn raw intracranial EEG recordings into a machine learning dataset.

I started by loading an OpenNeuro iEEG dataset, filtering the signal, removing bad channels, finding finger-tap events, and creating tap vs not-tap examples. My first logistic regression model used raw `channels x time` voltage windows and performed around chance level at about 49% accuracy.

This became the first useful failure in the project: it showed me that a classifier needs a better representation than thousands of raw voltage samples.

**Python skills:** MNE-Python, pandas, NumPy, scikit-learn, preprocessing, dataset creation  
**Neuroscience idea:** iEEG filtering, finger-tap events, epoching, and why raw voltage windows are hard to classify

The [linked project file](projects/ieeg-analysis-journey-first-tap-classifier.md) includes the preprocessing workflow, the first failed classifier, and what I learned from it.

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
- Familarise myself with machine learning concepts

---

## Connect

- GitHub: `https://github.com/YOUR_GITHUB_USERNAME`
- LinkedIn: `www.linkedin.com/in/sina-enayati-25b878328`
- Email: `senayati2006@gmail.com`

---

<div align="center">

### Learning Python by building tools for computational neuroscience.

</div>
