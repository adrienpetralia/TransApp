<h1 align="center">ADF & TransApp</h1>

<p align="center">
    <img width="600" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Intro.png" alt="Intro image">
</p>

<h1 align="center">A Transformer-Based Framework for Appliance Detection Using Smart Meter Consumption Series </h1>

### Data
The data used in this project comes from two sources:

<ul>
  <li>CER smart meter dataset from the ISSDA archive.</li>
  <li>Private smart meter dataset provide by EDF (Electricité De France).</li>
</ul> 

### Prerequisites 
To download the preprocessed subsample of the CER dataset (data/label/ExogneDate) :

<code>cd data</code>

Copy and paste the data from the link drive folder indicate in the README.md file.

Overall, the required python packages are listed as follows:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [imbalanced-learn](https://imbalanced-learn.org/stable/)
- [pytorch==1.8.1](https://pytorch.org/docs/1.8.1/)
- [torchinfo](https://pypi.org/project/torchinfo/0.0.1/)
- [scipy](https://scipy.org/)
- [sktime](http://www.sktime.net/en/latest/) 



### Appliance Detection Framework
Overview of our proposed Appliance Detection Framework.
<p align="center">
    <img width="800" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/Framework.png" alt="Framework image">
</p>

### TransApp Classifier
Overview of our proposed TransApp time series classifier.
<p align="center">
    <img width="700" src="https://github.com/adrienpetralia/TransApp/blob/master/ressources/all_model.png" alt="TransAppModel image">
</p>
