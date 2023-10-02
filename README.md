# augmentRNA

AugmentRNA is a simple toolbox for RNA-seq based datasets which is compatible both with Pandas and Polars
## Features
- Normalize data based on read counts
- Augment new samples of data for different labels based on negative binomial distributions or generative adversarial models, with tuneable noise
- Down-sample data data to equalize across class labels

## Installation

augmentRNA can be installed via the pip package manager for python

```sh
pip install augmentRNA
```

# Features
----------
# augment_data
```python
data = augment_data(data, num_samples, label, selected_label = 0, evals = False, epochs = 20, augment_type = 'nbinom', polars = False, normalize = False, noise = 0)
```
Augments new data samples for RNA-seq analysis

## Inputs
**data** : polars df, pandas df, str
    A dataframe containing the RNA-seq data, or a path to a .csv file of the dataframe

**num_samples** : int
    The additional numbers of samples that should be augmented from the data

**label** : str
    The label of the df column containing the classification label

**selected_label** : str, int
    The selected label that should be amplified. 'all' will amplify all labels to the selected amount

noise : float, int
    The amount of noise that should be applied to the data. A randomly selected value from the minimum and the maximumof the select gene column will be chosen them multiplied by the "noise" variable from -noise to noise, which will then be added to the data.

**augment_type** : str
    The type of augmentation that should be performed. A string containing 'nbinom' will sample from negative binomial  
    where applicable, otherwise sampling from a normal distribution, or for genes with no expression in the sample, will just output zeroes. A string containing 'gan' will sample from a generative adversarial network to generate samples.
    Defaults to nbinom

**evals** : bool
    Whether or not the mean squared error for each read count column should be calculated. Defaults to False

**polars** : bool
    Whether a polars (True) or pandas dataframe (False) should be used as the input. dataframe. Defaults to False

**normalize_data** : bool
    Whether the data should be normalized based on read counts. Defaults to False

**epochs** : int
    If a GAN is generated, how many epochs should the model be run for? Defaults to 20

## Outputs
**data** : polars df, pandas df
    Output dataframe containing augmented data and old data.
----------
# normalize_data
    data =  normalize_data(data, polars = False, round_data = True)
    
## Inputs

**data** : polars df, pandas df
    Input dataframe to normalize

**polars** : bool
    Whether a polars dataframe (True) or pandas dataframe (False) should be used

**round_data** : bool
    Whether the output values should be converted to integers or kept as floats

## Outputs

**data** : polars df, pandas df
    Output normalized dataframe
        
---------
# relevant_genes

    data = relevant_genes(data, label = 'RA', polars = False):
    
Filters dataset to only contain genes that have non-zero values in all columns, or zero vaues in all columns for every label.Seeks to minimize bias from different sequencing/sampling methods for different labels, and make the training dataset more representative.

## Inputs
**data** : polars df, pandas df, str
    RNA-seq expresison dataframe

**label** : str
    Dataframe column containing labels

**polars** : bool
    Whether pandas (False) or polars (True) dataframe is the input 

## Outputs
**data** : polars df, pandas df
    An output dataframe containing only genes that are relevant across all samples
    
## Development

This project is currently under active beta development. New features are being added, and if there is an additional processing feature that would fit the toolbox, please reach out the lead developer at *christian@defrondeville.com*.

## License

MIT
