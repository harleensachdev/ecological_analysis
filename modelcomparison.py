from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import warnings
from dataloading import BirdCallAnalyzer

# Initialize analyzer
analyzer = BirdCallAnalyzer()

# Load data
data = analyzer.load_data()

def generate_model_comparison_analysis(data, analyzer):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    count_cols = ['mean_alarm_count', 'mean_non_alarm_count', 'mean_background_count']

    # 4. Pairplot: Distribution relationships by model
    if len(data) > 1000:
        sample_data = data.sample(n=1000, random_state=42)
    else:
        sample_data = data
    
    g = sns.pairplot(sample_data[count_cols + ['model']], 
                     hue='model', diag_kind='kde', height=3)
    g.figure.suptitle('Count Distributions and Relationships by Model', y=1.02)
    
    analyzer.save_plot(g.figure, '13_pairplot_distributions.png')
    
# Run model comparison analysis
if data is not None:
    print("\nGenerating model comparison analysis...")
    generate_model_comparison_analysis(data, analyzer)
    print("Model comparison analysis complete!")


