import tqdm
import time
import pickle as pk
from pathlib import Path
import pandas as pd
import polars as pl
import sys
import random
from sklearn.utils import shuffle
import numpy as np
import statistics
from scipy.stats import nbinom
from multiprocessing.pool import ThreadPool as Pool

def add_noise(data, label = 'RA', noise = 0.1, noise_type = None, polars = True):
    '''
    Adds several different forms of noise to a dataframe of data. Ensures the end data is NOT negative (for RNAseq processing)
    
    Input
    --------------
    data: polars df, pandas df
        A data containing the data that should have noise injected
        
    label: str, int
        The label column name containing classes
        
    noise: float, int
        The proportion of noise added (by the proportion of the dataset mean/inddividual sample noise) to the data. Default .1
        
    noise_type: string
        The type of noise. Can either be mean (mean of the dataset noise) or uniform (across a uniform distribution of the maximum/minimum of the column). Defaults to uniform
        
    polars: bool
        Whether polars or pandas should be used. Defaults to True
    '''
    if isinstance(data, np.ndarray):
        X = data
    else:
        if polars == True:
            genes_to_keep = data.columns
            genes_to_keep.remove(label)
            X = data.drop(label).to_numpy()
            label = data.select([label])
        else:
            genes_to_keep = data.columns
            genes_to_keep.remove(label)
            X = data.drop(columns = [label]).to_numpy()
            label = data[label]

    data_length = X.shape[0]
    data_columns = X.shape[1]
    new_X = np.zeros((data_length, data_columns))
    if noise_type == None or noise_type.upper() == 'UNIFORM':
        for col in range(data_columns):
            column = X[:, col]
            max_val, min_val = max(column), min(column)
            for row in range(data_length):
                value = X[row, col]
                value = value + value * np.random.uniform(-1, 1) * noise
                if value < 0:
                    new_X[row, col] = 0
                else:
                    new_X[row, col] = value
                    
    elif noise_type.upper() == 'MEAN':
        mean = 0
        for col in range(X.shape[1]):
            column = X[:, col]
            mean += sum(column) / len(column)
        mean /= data_columns
        new_X = np.zeros((data_length, data_columns))
        for col in range(data_columns):
            column = X[:, col]
            for row in range(data_length):
                value = X[row, col]
                value = value + mean * np.random.uniform(-1, 1) * noise
                if value < 0:
                    new_X[row, col] = 0
                else:
                    new_X[row, col] = value

    if not isinstance(data, np.ndarray):
        # Adds labelt  back to data
        if polars == True:
            X = pl.DataFrame(new_X, schema = genes_to_keep)
            data = pl.concat([X, label], how = 'horizontal')
        else:
            X = pd.DataFrame(new_X, columns = genes_to_keep)
            data = pd.concat([X, label], axis = 1)
    else:
        data = X
        
    return data

def normalize_data(data, polars = False, round_data = True):
    '''
    Hormalizes data for read counts across different samples
    
    Inputs
    --------
    data : polars df, pandas df
        Input dataframe to normalize
    
    polars : bool
        Whether a polars dataframe (True) or pandas dataframe (False) should be used

    round_data : bool
        Whether the output values should be converted to integers or kept as floats

    Outputs
    --------
    data : polars df, pandas df
        Output normalized dataframe
    '''

    data_columns = data.columns

    if polars == True:
        read_counts = []
        for row in data.iter_rows():
            read_counts.append(sum(row[:-1]))
    
        read_mean = sum(read_counts)/len(read_counts)
        read_multipliers = [read/read_mean for read in read_counts]
        normal_data = []
        for num, row in enumerate(data.iter_rows()):
            multiplier = read_multipliers[num]
            normal = [int(i * multiplier) for i in row[:-1]] + [row[-1]]
            if round_data == True:
                normal = np.rint(normal).astype(int)
            normal_data.append(normal)
        data = pl.DataFrame(normal_data, schema = data_columns)
       
    else:
        read_counts = []
        for row in data.iterrows():
            read_counts.append(sum(list(row)[:-1]))
    
        read_mean = sum(read_counts)/len(read_counts)
        read_multipliers = [read/read_mean for read in read_counts]
        normal_data = []
        for num, row in enumerate(data.iterrows()):
            multiplier = read_multipliers[num]
            normal = [int(i * multiplier) for i in row[:-1]] + [row[-1]]
            if round_data == True:
                normal = np.rint(normal).astype(int)
            normal_data.append(normal)
        
        data = pd.DataFrame(normal_data, columns = data_columns, ignore_index = True)
   
    return data

def augment_data(data, num_samples, label = 'RA', selected_label = 0, epochs = 20,
                  augment_type = 'nbinom', polars = False, normalize = False, noise = 0):
    '''
    Augments new data samples for RNA-seq analysis

    Inputs
    -------------------------
    data : polars df, pandas df, str
        A dataframe containing the RNA-seq data, or a path to a .csv file of the dataframe

    num_samples : int
        The additional numbers of samples that should be augmented from the data

    label : str
        The label of the df column containing the classification label

    selected_label : str, int
        The selected label that should be amplified. 'all' will amplify all labels to the selected amount

    noise : float, int
        The amount of noise that should be applied to the data. A randomly selected value from the minimum and the maximum
        of the select gene column will be chosen them multiplied by the "noise" variable from -noise to noise, which will 
        then be added to the data.

    augment_type : str
        The type of augmentation that should be performed. A string containing 'nbinom' will sample from negative binomial  
        where applicable, otherwise sampling from a normal distribution, or for genes with no expression in the sample, will 
        just output zeroes. A string containing 'gan' will sample from a generative adversarial network to generate samples.
        Defaults to nbinom

    polars : bool
        Whether a polars (True) or pandas dataframe (False) should be used as the input dataframe. Defaults to False

    normalize_data : bool
        Whether the data should be normalized based on read counts. Defaults to False

    epochs : int
        If a GAN is generated, how many epochs should the model be run for? Defaults to 20

    Outputs
    ---------------
    data : polars df, pandas df
        Output dataframe containing augmented data and old data

    '''
    if polars == True:  
        try:
            data = pl.read_csv(data)
        except:
            pass

        if selected_label != 'all':
            selected_data = data.filter(pl.col(label) == selected_label)
            selected_data = selected_data.drop(label)
            length = len(selected_data)
        else:
            labels = data[label].to_list()
            label_counts = {}
            for l in labels:
                try:
                    label_counts[l] += 1
                except: 
                    label_counts[l] = 1
    else:
        try:
            data = pd.read_csv(data)
        except:
            pass

        if selected_label != 'all':
            selected_data = data[data[label] == selected_label].drop(label)
            length = len(selected_data)
        else:
            labels = list(data[label])
            label_counts = {}
            for l in labels:
                try:
                    label_counts[l] += 1
                except: 
                    label_counts[l] = 1
        
    data_columns = data.columns
    start = time.perf_counter()

    # If enabled, performs normalization of the data
    if normalize == True:
        data = normalize_data(data, round_data = False, polars = polars)
            
    if augment_type.upper() == 'NBINOM':
        def augment_genes(column, data = data, samples = num_samples):
            if column == label:
                return None
            exp = data[column]
            mean = exp.sum() / length
            if mean > 0:
                var = statistics.variance(exp)
                try:
                    k = (mean ** 2) / (var - mean)
                    if k <= 0:
                        std = statistics.stdev(exp)
                        generated_values = np.rint([i if i >= 0 else 0 for i in np.random.normal(mean, std, samples)]).astype(np.int64)
                        
                    else:
                        p = k / (k + mean)
                        generated_values = np.rint([i if i >= 0 else 0 for i in nbinom.rvs(n=k, p=p, size=samples)]).astype(np.int64)
                    
                except:
                    std = statistics.stdev(exp)
                    generated_values = np.rint([i if i >= 0 else 0 for i in np.random.normal(mean, std, samples)]).astype(np.int64)
                generated_values = list(generated_values)
            else:
                generated_values = [0 for _ in range(samples)]

            return generated_values

        if selected_label != 'all':
            augmented_data = {}
            for column in tqdm.tqdm(data_columns, total = len(data_columns), 
                                    desc = f'Augmenting {num_samples} samples for label {selected_label}'):
                augmented = augment_genes(column)
                if augmented != None:
                    augmented_data[column] = augmented

            if polars == True:
                augmented_labels = pl.DataFrame({label:[selected_label for _ in range(num_samples)]})
            else:
                augmented_labels = pd.DataFrame.from_dict({label:[selected_label for _ in range(num_samples)]})

        else:
            label_dict = {}
            unlabeled_columns = data_columns
            unlabeled_columns.remove(label)
            augmented_data = {i:[] for i in unlabeled_columns}
            sample_length = int(num_samples / len(list(set(label_counts.keys()))))
            for chosen_label in list(set(label_counts.keys())):
                remaining_samples =  sample_length - list(data[label]).count(chosen_label)
                label_counts[chosen_label] = remaining_samples
                label_dict[chosen_label] = remaining_samples

                if polars == True:
                    selected_data = data.filter(pl.col(label) == chosen_label).drop(label)
                else: 
                    selected_data = data[data[label] == chosen_label]
                    selected_data = selected_data.drop(columns = label)
                length = len(selected_data)
                for column in tqdm.tqdm(unlabeled_columns, total = len(unlabeled_columns),
                                         desc = f'Augmenting {remaining_samples} samples for label {chosen_label}'):
                    augmented = augment_genes(column, data = selected_data, samples = remaining_samples)
                    if augmented != None:
                        augmented_data[column] += augmented
        
            augmented_labels = [key for key, value in label_counts.items() for _ in range(value)]
        
            if polars == True:
                augmented_labels = pl.DataFrame({label:augmented_labels})
            else:
                augmented_labels = pd.DataFrame.from_dict({label:augmented_labels})
        
        augmented_data = {key:np.array(value).astype(np.int64) for key, value in augmented_data.items()}
       
        if polars == True:
            augmented_data = pl.concat([pl.DataFrame(augmented_data), augmented_labels], how = "horizontal")
            columns, aug_numpy = augmented_data.columns, np.rint(augmented_data.to_numpy())
            augmented_data = pl.DataFrame(aug_numpy, schema = columns)
            data_numpy = np.rint(data.to_numpy())
            data = pl.DataFrame(data_numpy, schema = columns)
            data = pl.concat([data, augmented_data], how = "vertical")

        else:
            augmented_data = pd.concat([
                pd.DataFrame(augmented_data), 
                augmented_labels
            ], axis=1, ignore_index=False).astype('int64')
            data = data.astype('int64')
            data = pd.concat([data, augmented_data], axis = 0)
        
    elif augment_type.upper() == 'GAN':
        try:
            from torch import nn
            import torch
            import torch.optim as optim
            from sklearn.preprocessing import StandardScaler, normalize
        except:
            print('GAN functionality requires torch and sklearn! Install them with pip install torch and pip install sklearn')
            sys.exit()

        # Define the Generator and Discriminator classes
        class Generator(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(Generator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                )
        
            def forward(self, x):
                return self.model(x)
        
        class Discriminator(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(Discriminator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_size * 10),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 10, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                )
        
            def forward(self, x):
                return self.model(x)
            
        if selected_label != 'all':
            X = selected_data.to_numpy()
            input_size = X.shape[1] 
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        else:
            selected_X = {}
            scalers = {}
            sample_numbers = {}
            labels = list(set(data[label]))
            sample_subset = int(num_samples/len(labels))

            if polars != True:
                
                for chosen_label in labels:
                    sample_numbers[chosen_label] = sample_subset - list(data[label]).count(chosen_label)
                    selected_X[chosen_label] =  data[data[label] == chosen_label].drop(label).to_numpy()
                    input_size = selected_X[chosen_label].shape[1] 
                    scaler = StandardScaler()
                    selected_X[chosen_label] = scaler.fit_transform(selected_X[chosen_label])
                    scalers[chosen_label] = scaler
                    
            else:
                for chosen_label in labels:
 
                    sample_numbers[chosen_label] = sample_subset - list(data[label]).count(chosen_label)
                    selected_data = data.filter(pl.col(label) == chosen_label)
                    selected_X[chosen_label] = selected_data.select([c for c in data.columns if c != label]).to_numpy()
                    input_size = selected_X[chosen_label].shape[1] 
                    scaler = StandardScaler()
                    selected_X[chosen_label] = scaler.fit_transform(selected_X[chosen_label])
                    scalers[chosen_label] = scaler

        hidden_size, output_size, batch_size = 512, 1, 64        
        G_lr, D_lr = 0.0002, 0.001               # Learning rate for the discriminator
        num_epochs = epochs          # Number of training epochs
        clip_value = 0.001           # Clip parameter for weight clipping (WGAN-specific)
        
        # Initialize networks
        generator = Generator(input_size, hidden_size, input_size)  # Output size matches input size
        discriminator = Discriminator(input_size, hidden_size, output_size)
        
        # Loss function and optimizers
        optimizer_G = optim.RMSprop(generator.parameters(), lr=G_lr, weight_decay = .001)
        optimizer_D = optim.RMSprop(discriminator.parameters(), lr=D_lr, weight_decay = .001)
        
        def run_model(X, input_scaler, samples = num_samples, chosen_label = selected_label):
            for epoch in tqdm.tqdm(range(num_epochs), total = num_epochs, desc = f'Training GAN to generate {samples} samples of label {chosen_label}'):
                for i in range(0, X.shape[0], batch_size):
                    # Sample real data
                    real_data = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                    
                    # Sample noise for generator
                    gen_noise = torch.randn(batch_size, input_size)
                    
                    # Generate fake data from noise
                    fake_data = generator(gen_noise)
                    
                    # Discriminator forward and backward pass
                    optimizer_D.zero_grad()
                    
                    # Compute the discriminator scores for real and fake data
                    real_scores = discriminator(real_data)
                    fake_scores = discriminator(fake_data)
                    
                    # Compute the Wasserstein loss
                    loss_D = -torch.mean(real_scores) + torch.mean(fake_scores)
                    loss_D.backward()
                    
                    fake_data, real_scores, fake_scores = fake_data.detach(), real_scores.detach(), fake_scores.detach()
                
                    # Weight clipping (WGAN-specific)
                    for param in discriminator.parameters():
                        param.data.clamp_(-clip_value, clip_value)
                    
                    # Update discriminator
                    optimizer_D.step()
                    
                    # Generator forward and backward pass
                    optimizer_G.zero_grad()
                
                    # Compute the discriminator scores for fake data (detach to avoid backpropagation)
                    fake_scores = discriminator(fake_data)
                
                    # Compute the generator loss
                    loss_G = -torch.mean(fake_scores)
                
                    loss_G.backward()
                
                    # Update generator
                    optimizer_G.step()
                
                # Print progress
                if epoch == num_epochs - 1:
                    print(f"Wasserstein Loss (D): {loss_D.item():.4f}, Wasserstein Loss (G): {loss_G.item():.4f}")
            
            # Generate fake samples
            generated_noise = torch.randn(samples, input_size)
            faked_samples = input_scaler.inverse_transform(generator(generated_noise).detach().numpy())
        
            labels = np.array([chosen_label for _ in range(samples)]).reshape(-1, 1)
    
            fake_samples = np.hstack((np.rint(faked_samples), labels))

            return fake_samples
        
        if selected_label != 'all':
            fake_samples = run_model(X, input_scaler = scaler)
        else:
            for chosen_label in labels:
                scaler = scalers[chosen_label]
                X = selected_X[chosen_label]
                faked = run_model(X, input_scaler = scaler, chosen_label = chosen_label, samples = sample_numbers[chosen_label] )
        
                try:
                    fake_samples = np.vstack((fake_samples, faked))
                except:
                    fake_samples = faked

        if polars == True:
            fake_samples = pl.DataFrame(fake_samples, schema = data_columns)
            data_numpy = np.rint(data.to_numpy())
            data = pl.DataFrame(data_numpy, schema = data_columns)
            data = pl.concat((data, fake_samples), how = "vertical")
        else:
            fake_samples = pd.DataFrame(fake_samples, columns = data_columns)
            data = data.astype('int64')
            data = pd.concat([data, fake_samples], axis = 0)
            
    if noise > 0:
        data = add_noise(data, noise = noise)
        
    end = time.perf_counter()
    print(f'Data augmented to {num_samples} samples in {round(end - start, 4)} seconds')

    return data

def relevant_genes(data, label = 'RA', polars = False):
    '''
    Filters dataset to only contain genes that have non-zero values in all columns, or zero vaues in all columns for every label.
    Seeks to minimize bias from different sequencing/sampling methods for different labels, and make the training dataset more representative.
    
    Inputs
    -------------
    data : polars df, pandas df, str
        RNA-seq expresison dataframe

    label : str
        Dataframe column containing labels

    polars : bool
        Whether pandas (False) or polars (True) dataframe is the input 

    Outputs
    ---------------
    data : polars df, pandas df
        An output dataframe containing only genes that are relevant across all samples
    '''
    try:
        if polars == True:
            data = pl.read_csv(data)
        else:
            data = pd.read_csv(data)
    except:
        pass

    selected_genes = []
    genes = data.columns
    mean_statistics = []
   
    labels = list(set(data[label]))
    label_length = len(labels)

    for count, chosen_label in enumerate(labels):
        if polars == True:
            subset = data.filter(pl.col(label) == chosen_label)
        else:
            subset = data[data[label] == chosen_label]

        for num, column in enumerate(genes):
            mean = sum(subset[column]/len(subset[column]))
            if count == 0:
                mean_statistics.append([mean])
            else:
                mean_statistics[num].append(mean)
               
    for count, stat in enumerate(mean_statistics):

        if 0 in stat:
            if stat.count(0) == label_length:
                selected_genes.append(genes[count])
        else:
            selected_genes.append(genes[count])
        
    selected_genes += [label]
    if polars != True:
        data = data[selected_genes] 
    else:
        data = data.select(selected_genes)
  
    print(f'{len(mean_statistics)} trimmed down to {len(data.columns)} relevant genes')
    return data

if __name__ == '__main__':
    data = relevant_genes(Path('/work/ccnr/GeneFormer/GeneFormer_repo/Enzo_dataset.csv'), polars = True)
    data = augment_data(data, polars = True,
                        num_samples = 4200, selected_label = 0, normalize = False, augment_type = 'GAN')
                        