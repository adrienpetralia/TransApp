<h1 align="center">ADF & TransApp</h1>

<p align="center">
    <img width="450" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Intro.png" alt="Intro image">
</p>

<h2 align="center">A Transformer-Based Framework for Appliance Detection Using Smart Meter Consumption Series (VLDB 2024)</h2>

<div align="center">
<p>
<img alt="GitHub issues" src="https://img.shields.io/github/issues/adrienpetralia/TransApp">
</p>
</div>

#### ✨ **News:** This work has been accepted for publication in the [Proceedings of the VLDB Endowment](https://www.vldb.org/pvldb/volumes/17) and will appear in the [50th International Conference on Very Large Databases (VLDB 2024)](https://vldb.org/2024/).

## References
> Adrien Petralia, Philippe Charpentier, and Themis Palpanas. ADF & TransApp:
> A Transformer-Based Framework for Appliance Detection Using Smart Meter Consumption Series.
> Proceedings of the VLDB Endowment (PVLDB), 17(3): 553 - 562, 2023.
> doi:10.14778/3632093.363211


## Proposed approach

### Appliance Detection Framework
We propose the Appliance Detection Framework (ADF) to detect the presence of appliances in households, using real-world consumption series, which are sampled at a very low frequency, and are long and variable-length. ADF addresses these challenges by operating at individual subsequences of each consumption series, instead of each series in its entirety. The framework can be used with any time series classifier designed to predict probabilities.
<p align="center">
    <img width="650" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Framework.png" alt="Framework image">
</p>

### TransApp
We propose TransApp, a Transformer-based time series classifier, which can first be pretrained in a self-supervised manner to enhance its ability on appliances detection tasks. This way, TransApp can significantly improve its accuracy.

#### Model architecture
The proposed architecture lies in combination of a strong embedding block made of dilated convolutional layers followed by a Transformer encoder using Diagonally Masked Self-Attention (DMSA).
<p align="center">
    <img width="650" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/all_model.png" alt="TransAppModel image">
</p>

#### Two steps training process
**Self-supervised pretraining.** The use of a self-supervised pretraining of a Transformer architecture on an auxiliary task has been used in the past to boost the model performance on downstream tasks. This process is inspired by the mask-based pretraining of vision transformer and requires only the input consumption series without any appliance information label. It results in a reconstruction objective of a corrupted (masked) time series fed to the model input.

**Supervised pretraining.** The supervised training results in a simple binary classification process using labeled time series.

<p align="center">
    <img width="400" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Training.png" alt="Two-steps training image">
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


## Contributors

* [Adrien Petralia](https://adrienpetralia.github.io/), EDF R&D, Université Paris Cité
* [Philippe Charpentier](https://www.researchgate.net/profile/Philippe-Charpentier), EDF R&D
* [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/), Université Paris Cité, IUF

## Acknowledgments

We would like to thanks [Paul Boniol](https://boniolp.github.io/) for the valuable discussions on this project.
Work supported by EDF R&D and ANRT French program.
