<h1 align="center">ADF & TransApp</h1>

<p align="center">
    <img width="450" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Intro.png" alt="Intro image">
</p>

<h2 align="center">A Transformer-Based Framework for Appliance Detection Using Smart Meter Consumption Series </h2>

<div align="center">
<p>
<img alt="GitHub issues" src="https://img.shields.io/github/issues/adrienpetralia/TransApp">
</p>
</div>

#### ✨ **News:** This work has been accepted for publication in the [50th International Conference on Very Large Databases (VLDB 2024)](https://vldb.org/2024/).

## References
> Adrien Petralia, Philippe Charpentier, Themis Palpanas. ADF & TransApp:
> A Transformer-Based Framework for Appliance Detection Using Smart Meter Consumption Series.
> Proceedings of the VLDB Endowment (PVLDB) Journal, 2024

## Getting Started

### Prerequisites 

Python version : <code> >= Python 3.7 </code>

Overall, the required python packages are listed as follows:

<ul>
    <li><a href="https://numpy.org/">numpy</a></li>
    <li><a href="https://pandas.pydata.org/">pandas</a></li>
    <li><a href="https://scikit-learn.org/stable/">scikit-learn</a></li>
    <li><a href="https://imbalanced-learn.org/stable/">imbalanced-learn</a></li>
    <li><a href="https://pytorch.org/docs/1.8.1/">torch==1.8.1</a></li>
    <li><a href="https://pypi.org/project/torchinfo/0.0.1/">torchinfo</a></li>
    <li><a href="https://scipy.org/">scipy</a></li>
    <li><a href="http://www.sktime.net/en/latest/">sktime</a></li>
    <li><a href="https://matplotlib.org/">matplotlib</a></li>
</ul>

### Installation

Use pip to install all the required libraries listed in the requirements.txt file.

```
pip install -r requirements.txt
```

### Data
The data used in this project comes from two sources:

<ul>
  <li>CER smart meter dataset from the ISSDA archive.</li>
  <li>Private smart meter dataset provide by EDF (Electricité De France).</li>
</ul> 

You may find more information on how to access the datasets in the [data](https://github.com/adrienpetralia/TransApp/tree/main/data) folder.

## Architecture Overview

### Appliance Detection Framework
Overview of our proposed Appliance Detection Framework.
<p align="center">
    <img width="650" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Framework.png" alt="Framework image">
</p>

### TransApp architecture
Overview of our proposed TransApp time series classifier.
<p align="center">
    <img width="600" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/all_model.png" alt="TransAppModel image">
</p>

### Two steps training process architecture
Improving appliance quality detection with a pretraining step using non-labeled data.
<p align="center">
    <img width="600" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Training.png" alt="Two-steps training image">
</p>

## Experiments

We provide a jupyter-notebook example to use our Appliance Detection Framework combined with our TransApp classifier on the CER data : experiments/[TransAppExample.ipynb](https://github.com/adrienpetralia/TransApp/tree/main/experiments/TransAppExample.ipynb). 

In addition, to reproduce papers experiments, use the following guidelines.

### Pretraining TransApp

Pretraining TransApp in a self-supervised way using non labeled data :

```
sh LaunchTransAppPretraining.sh
```

### Appliance Detection with TransApp

Use our Appliance Detection Framework combined with TransApp to detect appliance in consumption time series :

```
sh LaunchTransAppClassif.sh
```

### Appliance Detection with other time series classifiers

#### Inside our Appliance Detection Framework

Use our Appliance Detection Framework combined with ConvNet, ResNet or InceptionTime to detect appliance in consumption time series :

```
sh LaunchModelsClassif.sh
```

#### Outside our Appliance Detection Framework

Please refer to this Github [ApplianceDetectionBenchmark](https://github.com/adrienpetralia/ApplianceDetectionBenchmark) to reproduce the experiments, where an extensive evaluation of different time series classifiers have been conducted, inluding on the datasets used in this study.

