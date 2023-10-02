=======================
augmentRNA Documentation
=======================

.. image:: https://img.shields.io/pypi/v/augmentRNA.svg
        :target: https://pypi.python.org/pypi/augmentRNA

.. image:: https://img.shields.io/travis/username/augmentRNA.svg
        :target: https://travis-ci.org/username/augmentRNA

AugmentRNA is a straightforward toolbox designed for RNA-seq based datasets that is compatible with both Pandas and Polars libraries.

Features
--------

- Normalize data based on read counts
- Augment new samples of data for different labels based on negative binomial distributions or generative adversarial models, with tunable noise
- Down-sample data to equalize across class labels

Installation
------------

augmentRNA can be installed via the pip package manager for Python.

.. code-block:: sh

    pip install augmentRNA

augment_data
------------

.. code-block:: python

    data = augment_data(data, num_samples, label, selected_label=0, evals=False, epochs=20, augment_type='nbinom', polars=False, normalize=False, noise=0)

Augments new data samples for RNA-seq analysis.

Inputs:

- **data**: Polars df, Pandas df, str
  A dataframe containing the RNA-seq data, or a path to a .csv file of the dataframe.
- **num_samples**: int
  The additional number of samples that should be augmented from the data.
- **label**: str
  The label of the df column containing the classification label.
- **selected_label**: str, int
  The selected label that should be amplified. 'all' will amplify all labels to the selected amount.
- **noise**: float, int
  The amount of noise that should be applied to the data. A randomly selected value from the minimum and the maximum of the select gene column will be chosen then multiplied by the "noise" variable from -noise to noise, which will then be added to the data.
- **augment_type**: str
  The type of augmentation that should be performed. A string containing 'nbinom' will sample from the negative binomial where applicable, otherwise sampling from a normal distribution, or for genes with no expression in the sample, will just output zeroes. A string containing 'gan' will sample from a generative adversarial network to generate samples. Defaults to nbinom.
- **evals**: bool
  Whether or not the mean squared error for each read count column should be calculated. Defaults to False.
- **polars**: bool
  Whether a Polars (True) or Pandas dataframe (False) should be used as the input. Defaults to False.
- **normalize_data**: bool
  Whether the data should be normalized based on read counts. Defaults to False.
- **epochs**: int
  If a GAN is generated, how many epochs should the model be run for? Defaults to 20.

Outputs:

- **data**: Polars df, Pandas df
  Output dataframe containing augmented data and old data.

normalize_data
--------------

.. code-block:: python

    data = normalize_data(data, polars=False, round_data=True)

Inputs:

- **data**: Polars df, Pandas df
  Input dataframe to normalize.
- **polars**: bool
  Whether a Polars dataframe (True) or Pandas dataframe (False) should be used.
- **round_data**: bool
  Whether the output values should be converted to integers or kept as floats.

Outputs:

- **data**: Polars df, Pandas df
  Output normalized dataframe.

relevant_genes
--------------

.. code-block:: python

    data = relevant_genes(data, label='RA', polars=False)

Filters the dataset to only contain genes that have non-zero values in all columns, or zero values in all columns for every label. Seeks to minimize bias from different sequencing/sampling methods for different labels, and make the training dataset more representative.

Inputs:

- **data**: Polars df, Pandas df, str
  RNA-seq expression dataframe.
- **label**: str
  Dataframe column containing labels.
- **polars**: bool
  Whether Pandas (False) or Polars (True) dataframe is the input.

Outputs:

- **data**: Polars df, Pandas df
  An output dataframe containing only genes that are relevant across all samples.

Development
-----------

This project is currently under active beta development. New features are being added, and if there is an additional processing feature that would fit the toolbox, please reach out to the lead developer at *christian@defrondeville.com*.

License
-------

MIT
