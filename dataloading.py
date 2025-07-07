import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class BirdCallAnalyzer:
    """ Loading csv files and extracting model and site names"""
    def __init__(self, data_path=".", output_folder="analysis_results"):
        self.data_path = Path(data_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.data = None
        
    
    def load_data(self, specific_files=None):
        """
        Load CSV files either from specific paths or pattern matching
        
        Args:
            specific_files: List of specific file paths to load
        """
        files_to_load = ['/Users/caramelloveschicken/Desktop/data/National Park/NP-Results/np-2-fs1-results-aggregated.csv','/Users/caramelloveschicken/Desktop/data/National Park/NP-Results/np-2-fs2-results-aggregated.csv', '/Users/caramelloveschicken/Desktop/data/National Park/NP-Results/np-2-fs3-results-aggregated.csv', '/Users/caramelloveschicken/Desktop/data/Recreational Park/TB001-Results/tb-1-fs1-results-aggregated.csv','/Users/caramelloveschicken/Desktop/data/Recreational Park/TB001-Results/tb-1-fs2-results-aggregated.csv','/Users/caramelloveschicken/Desktop/data/Recreational Park/TB001-Results/tb-1-fs3-results-aggregated.csv','/Users/caramelloveschicken/Desktop/data/Habitat Park/PH-Results/ph-1-fs1-results-aggregated.csv','/Users/caramelloveschicken/Desktop/data/Habitat Park/PH-Results/ph-1-fs2-results-aggregated.csv','/Users/caramelloveschicken/Desktop/data/Habitat Park/PH-Results/ph-1-fs3-results-aggregated.csv' ]
        
        # Option 1: Use files passed to this method
        if specific_files:
            files_to_load = [Path(f) for f in specific_files]
        
        
            
        all_data = []
        
        for file in files_to_load:
            # Extract site and model from filename
            file = Path(file)
            filename = file.stem
            parts = filename.split('-')
            if len(parts) >= 3:
                site = f"{parts[0]}-{parts[1]}"  # e.g., "tb-1"
                model_part = parts[2]  # e.g., "fs1"
                model = model_part
            else:
                continue
                
            df = pd.read_csv(file)
            df['site'] = site
            df['model'] = model
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            all_data.append(df)
            print(f"Loaded {len(df)} records from {file.name}")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal records loaded: {len(self.data)}")
            print(f"Sites: {sorted(self.data['site'].unique())}")
            print(f"Models: {sorted(self.data['model'].unique())}")
            return self.data
        return None
    
    def save_plot(self, fig, filename, dpi=300):
        """Save plot to output folder"""
        filepath = self.output_folder / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

# Initialize analyzer
analyzer = BirdCallAnalyzer()

# Load data
data = analyzer.load_data()

if data is not None:
    print("\nData loaded successfully!")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"\nSample data structure:")
    print(data.head())
else:
    print("Failed to load data. Please check file paths and naming convention.")


    