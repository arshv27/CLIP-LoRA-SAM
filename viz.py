import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('results3.csv', names=['dataset', 'shots', 'seed', 'rho', 'accuracy'])

# Parse the required values
datasets = df['dataset'].unique()

# Define the additional mean accuracy values and standard deviations
CLIP_LORA_CALTECH_MEAN = [93.7, 95.2, 96.4]
CLIP_LORA_CALTECH_STD = [0, 0, 0]

CLIP_LORA_FGVC_MEAN = [30.2, 37.9, 54.7]
CLIP_LORA_FGVC_STD = [0, 0, 0]

CLIP_LORA_EUROSAT_MEAN = [72.3, 84.9, 92.1]
CLIP_LORA_EUROSAT_STD = [0, 0, 0]

CLIP_LORA_IMAGENET_MEAN = [70.4, 71.4, 73.6]
CLIP_LORA_IMAGENET_STD = [0, 0, 0]

# Create plots for each unique dataset
for dataset in datasets:
    print(dataset)
    df_dataset = df[df['dataset'] == dataset]
    rhos = df_dataset['rho'].unique()
    
    plt.figure(figsize=(10, 6))
    
    for rho in rhos:
        df_rho = df_dataset[df_dataset['rho'] == rho]
        shots = df_rho['shots'].unique()
        
        means = []
        stds = []
        
        for shot in shots:
            df_shot = df_rho[df_rho['shots'] == shot]
            accuracies = df_shot['accuracy']
            means.append(accuracies.mean())
            stds.append(accuracies.std())
        
        means = np.array(means)
        stds = np.array(stds)
        
        plt.plot(shots, means, label=f'rho={rho}')
        # plt.fill_between(shots, means - stds, means + stds, alpha=0.2)
        plt.scatter(shots, means)
        
    if dataset == 'caltech101':
        plt.plot(shots, CLIP_LORA_CALTECH_MEAN, label='W/O SAM', linestyle='--')
        plt.scatter(shots, CLIP_LORA_CALTECH_MEAN)
    elif dataset == 'fgvc':
        plt.plot(shots, CLIP_LORA_FGVC_MEAN, label='W/O SAM', linestyle='--')
        plt.scatter(shots, CLIP_LORA_FGVC_MEAN)
    elif dataset == 'eurosat':
        plt.plot(shots, CLIP_LORA_EUROSAT_MEAN, label='W/O SAM', linestyle='--')
        plt.scatter(shots, CLIP_LORA_EUROSAT_MEAN)
    elif dataset == 'imagenet':
        plt.plot(shots, CLIP_LORA_IMAGENET_MEAN, label='W/O SAM', linestyle='--')
        plt.scatter(shots, CLIP_LORA_IMAGENET_MEAN)
    
    plt.title(f'Accuracy vs Shots for {dataset} (A-SAM)')
    plt.xlabel('Shots')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'accuracy_vs_shots_{dataset}_A-SAM.png', dpi=300, bbox_inches='tight')