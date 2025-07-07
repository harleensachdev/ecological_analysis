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
plt.style.use('default')
sns.set_palette("husl")

# Define metrics and their display names
METRICS = ['mean_alarm_count', 'mean_non_alarm_count', 'mean_background_count']
METRIC_NAMES = ['Alarm Count', 'Non-Alarm Count', 'Background Count']
METRIC_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue


# 1. Monthly Performance - All labels
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

monthly_data = data.groupby(['site', 'model', 'month'])[METRICS].mean().reset_index()

for i, (metric, metric_name, color) in enumerate(zip(METRICS, METRIC_NAMES, METRIC_COLORS)):
    ax = axes[i]
    
    for site in sorted(data['site'].unique()):
        site_data = monthly_data[monthly_data['site'] == site]
        for model in sorted(data['model'].unique()):
            model_data = site_data[site_data['model'] == model]
            ax.plot(model_data['month'], model_data[metric], 
                    marker='o', label=f'{site}-{model}', linewidth=2)
    
    ax.set_title(f'Monthly {metric_name} Trends', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel(f'Mean {metric_name}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F8F9FA')

plt.tight_layout()
analyzer.save_plot(fig, '07_monthly_trends_all_labels.png')

# 2. Faceted monthly plots - All labels
sites = sorted(data['site'].unique())
models = sorted(data['model'].unique())

# Create separate plots for each metric
for metric_idx, (metric, metric_name, base_color) in enumerate(zip(METRICS, METRIC_NAMES, METRIC_COLORS)):
    fig, axes = plt.subplots(1, len(sites), figsize=(6*len(sites), 5))
    if len(sites) == 1:
        axes = [axes]
    
    # Create color palette for models
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    for i, site in enumerate(sites):
        ax = axes[i]
        site_data = monthly_data[monthly_data['site'] == site]
        
        for model in models:
            model_data = site_data[site_data['model'] == model]
            if len(model_data) > 0:
                ax.plot(model_data['month'], model_data[metric],
                        marker='o', label=model, linewidth=2, 
                        color=model_colors[model], markersize=6)
        
        ax.set_title(f'{metric_name} - Site: {site}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel(f'Mean {metric_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Model', loc='best', frameon=True, fancybox=True, shadow=True)
        ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    analyzer.save_plot(fig, f'08_monthly_faceted_{metric}.png')

# 3. Hourly/Diurnal Patterns - All labels
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

hourly_data = data.groupby(['site', 'model', 'hour'])[METRICS].mean().reset_index()

for i, (metric, metric_name, color) in enumerate(zip(METRICS, METRIC_NAMES, METRIC_COLORS)):
    ax = axes[i]
    
    for site in sorted(data['site'].unique()):
        site_data = hourly_data[hourly_data['site'] == site]
        for model in sorted(data['model'].unique()):
            model_data = site_data[site_data['model'] == model]
            ax.plot(model_data['hour'], model_data[metric], 
                    marker='o', label=f'{site}-{model}', linewidth=2, markersize=4)
    
    ax.set_title(f'Hourly {metric_name} Patterns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(f'Mean {metric_name}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    ax.set_facecolor('#F8F9FA')

plt.tight_layout()
analyzer.save_plot(fig, '09_hourly_patterns_all_labels.png')

# 4. Radial plots for 24-hour patterns - All labels
for metric_idx, (metric, metric_name, base_color) in enumerate(zip(METRICS, METRIC_NAMES, METRIC_COLORS)):
    fig, axes = plt.subplots(1, len(sites), figsize=(6*len(sites), 6), subplot_kw=dict(projection='polar'))
    if len(sites) == 1:
        axes = [axes]
    
    for i, site in enumerate(sites):
        ax = axes[i]
        site_hourly = hourly_data[hourly_data['site'] == site]
        
        for model in sorted(data['model'].unique()):
            model_data = site_hourly[site_hourly['model'] == model]
            
            if len(model_data) > 0:
                # Convert hour to radians
                theta = np.array(model_data['hour']) * 2 * np.pi / 24
                r = model_data[metric]
                
                ax.plot(theta, r, marker='o', label=model, linewidth=2)
        
        ax.set_title(f'24h {metric_name} Pattern - {site}', pad=20, fontweight='bold')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 360, 30), 
                            labels=[f'{h:02d}:00' for h in range(0, 24, 2)])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    plt.tight_layout()
    analyzer.save_plot(fig, f'10_radial_hourly_{metric}.png')
    
# 5: Hourly mean counts for each label across each site (by time segment)
# adding the hour column
df = analyzer.load_data()

df["hour"] = df["datetime"].dt.hour 

hourly = (df
            .groupby(["site", "model", "hour"])[["mean_alarm_count", "mean_non_alarm_count", "mean_background_count"]]
            .mean()
            .reset_index()
        )

# grid
metrics = ['mean_alarm_count', 'mean_non_alarm_count', 'mean_background_count']
titles  = ['Alarm calls', 'Non-alarm', 'Background']

for metric, title in zip(metrics, titles):
    g = sns.FacetGrid(hourly, col = "site", hue = "model", col_wrap = 3, height = 4, sharey = False, palette="Set2")
    g.map(sns.lineplot, "hour", metric, marker = "o")
    g.set_titles("{col_name}")
    g.add_legend(title="Model")      
    g.set_axis_labels("Hour of day", f'Mean {title} predictions')
    g.figure.suptitle(f"Hourly {title}-call predictions by model across sites", y=1.04)
    analyzer.save_plot(g.figure, f"{title} Call hourly_predictions_faceted.png")


    heat = (
        df.groupby(["site", "hour"])[metric]
        .mean().unstack(level = 0))

    plt.figure(figsize=(12,4))
    sns.heatmap(heat, cmap="mako", annot = False)
    plt.title(f"Average {title} call probability by hour and by site")
    plt.xlabel("Site")
    plt.ylabel("Hour")
    plt.tight_layout()
    analyzer.save_plot(plt.gcf(), f"{title}_Calls_hourly_heatmap.png")

    # hourly analysis to compare sites

    hourly = (df
                .groupby(["site", "model", "hour"])[["mean_alarm_count", "mean_non_alarm_count", "mean_background_count"]]
                .mean()
                .reset_index()
            )

    # grid
    g = sns.FacetGrid(hourly, col = "model", hue = "site", col_wrap = 3, height = 4, sharey = False, palette="Set2")
    g.map(sns.lineplot, "hour", metric, marker = "o")
    g.set_titles("{col_name}")
    g.add_legend(title = "Site")
    g.set_axis_labels("Hour of day", f"Mean {title} predictions")
    g.figure.suptitle(f"Hourly {title} call predictions by model across sites", y=1.04)
    analyzer.save_plot(g.figure, f"{title}_model_across_site_faceted.png")

    # reset index converts this to regular columns

# Initialize analyzer
analyzer = BirdCallAnalyzer()

# Load data
data = analyzer.load_data()

# 6: Weekly analysis by days of the week
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

data['day_of_week'] = data['datetime'].dt.day_name()
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
data['day_of_week'] = pd.Categorical(data['day_of_week'], categories=dow_order, ordered=True)

weekly_data = (
data.groupby(['site', 'model', 'day_of_week'])
[['mean_alarm_count', 'mean_non_alarm_count', 'mean_background_count']]
.mean()
.reset_index()
)
# Line plots for each metric
metrics = ['mean_alarm_count', 'mean_non_alarm_count', 'mean_background_count']
titles = ['Alarm Count', 'Non-Alarm Count', 'Background Count']


for site in sorted(data['site'].unique()):
    fig, axes = plt.subplots(1,3, figsize= (15,4), sharey = False)
    site_df = weekly_data[weekly_data['site'] == site]
        
    for ax, metric, title in zip(axes, metrics, titles):
        sns.lineplot(
            data = site_df,
            x    = 'day_of_week',
            y    = metric,
            hue  = 'model',           # one curve per model
            marker='o',
            ax   = ax
        )
        ax.set_title(f'{site} – {title} by Day of Week')
        ax.set_xlabel('')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Model', loc='best')

    plt.tight_layout()
    analyzer.save_plot(fig, f'{site}_dayofweek_temporal.png')
    plt.close(fig)


# Seasonal analysis

def prepare_temporal_features_malaysia(data):
    """Add temporal features specific to Malaysian climate and bird patterns"""
    data = data.copy()
    
    # Basic temporal features
    data['day_of_week'] = data['datetime'].dt.day_name()
    data['day_of_week_num'] = data['datetime'].dt.dayofweek
    data['is_weekend'] = data['day_of_week_num'].isin([5, 6])
    data['week_of_year'] = data['datetime'].dt.isocalendar().week
    data['month_name'] = data['datetime'].dt.month_name()

    
    # Malaysian bird activity periods
    hour = data['datetime'].dt.hour
    data['activity_period'] = 'Day'
    data.loc[hour.isin([5, 6, 7, 8]), 'activity_period'] = 'Morning Chorus'
    data.loc[hour.isin([17, 18, 19]), 'activity_period'] = 'Evening Chorus'
    data.loc[hour.isin([20, 21, 22, 23, 0, 1, 2, 3, 4]), 'activity_period'] = 'Night'
    data.loc[hour.isin([12, 13, 14, 15]), 'activity_period'] = 'Midday Quiet'

    return data


def generate_tropical_daily_patterns_all_labels(data, analyzer):
    """Daily activity patterns for all labels specific to tropical bird behavior"""
    print("Generating tropical daily activity patterns for all labels...")
    
    # Create comprehensive daily patterns for each metric
    for metric_idx, (metric, metric_name, color) in enumerate(zip(METRICS, METRIC_NAMES, METRIC_COLORS)):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{metric_name} - Tropical Daily Activity Patterns', fontsize=16, fontweight='bold')
        
        # 2. Hourly patterns with tropical context
        hourly_data = data.groupby(['site', 'model', 'datetime'])[metric].mean().reset_index()
        hourly_data['hour'] = hourly_data['datetime'].dt.hour
        hourly_avg = hourly_data.groupby(['site', 'model', 'hour'])[metric].mean().reset_index()
        
        for site in sorted(data['site'].unique()):
            site_data = hourly_avg[hourly_avg['site'] == site]
            for model in sorted(data['model'].unique()):
                model_data = site_data[site_data['model'] == model]
                axes[0,1].plot(model_data['hour'], model_data[metric], 
                              marker='o', label=f'{site}-{model}', linewidth=2)
        
        axes[0,1].set_title(f'24-Hour {metric_name} Pattern')
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel(f'Mean {metric_name}')
        axes[0,1].set_xticks(range(0, 24, 2))
        axes[0,1].axvspan(5, 8, alpha=0.2, color='gold', label='Morning Chorus')
        axes[0,1].axvspan(17, 19, alpha=0.2, color='orange', label='Evening Chorus')
        axes[0,1].axvspan(12, 15, alpha=0.2, color='lightgray', label='Midday Quiet')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].grid(True, alpha=0.3)
    
        
        # 4. Weekend vs weekday patterns
        weekend_data = data.groupby(['site', 'model', 'is_weekend', 'datetime'])[metric].mean().reset_index()
        weekend_data['hour'] = weekend_data['datetime'].dt.hour
        weekend_hourly = weekend_data.groupby(['site', 'model', 'is_weekend', 'hour'])[metric].mean().reset_index()
        
        for site in sorted(data['site'].unique()):
            site_data = weekend_hourly[weekend_hourly['site'] == site]
            weekday_data = site_data[site_data['is_weekend'] == False].groupby('hour')[metric].mean()
            weekend_data_plot = site_data[site_data['is_weekend'] == True].groupby('hour')[metric].mean()
            
            axes[1,1].plot(weekday_data.index, weekday_data.values, 
                          label=f'{site} Weekday', linestyle='-', marker='o')
            axes[1,1].plot(weekend_data_plot.index, weekend_data_plot.values, 
                          label=f'{site} Weekend', linestyle='--', marker='s')
        
        axes[1,1].set_title('Weekday vs Weekend Activity Patterns')
        axes[1,1].set_xlabel('Hour of Day')
        axes[1,1].set_ylabel(f'Mean {metric_name}')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        analyzer.save_plot(fig, f'11_tropical_daily_{metric}.png')

data_enhanced = prepare_temporal_features_malaysia(data)

generate_tropical_daily_patterns_all_labels(data_enhanced, analyzer)

def generate_comprehensive_summary_all_labels(data, analyzer):
    """Generate comprehensive summary statistics for all labels"""
    print("Generating comprehensive summary for all labels...")
    
    # Overall statistics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Bird Call Analysis Summary - All Labels', fontsize=16, fontweight='bold')
    
    # 1. Overall activity distribution
    activity_data = data[METRICS].melt(var_name='Metric', value_name='Count')
    sns.boxplot(data=activity_data, x='Metric', y='Count', ax=axes[0,0])
    axes[0,0].set_title('Overall Activity Distribution')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Site-wise activity summary
    site_summary = data.groupby('site')[METRICS].mean()
    site_summary.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Average Activity by Site')
    axes[0,1].legend(title='Activity Type')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Model performance summary
    model_summary = data.groupby('model')[METRICS].mean()
    model_summary.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average Activity by Model')
    axes[1,0].legend(title='Activity Type')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Temporal summary
    temporal_summary = data.groupby('activity_period')[METRICS].mean()
    temporal_summary.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Average Activity by Time Period')
    axes[1,1].legend(title='Activity Type')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    analyzer.save_plot(fig, '14_comprehensive_summary.png')


def generate_weekend_weekday_hourly_analysis(data, analyzer):
    """Generate weekend vs weekday hourly patterns for each label at each site"""
    print("Generating weekend vs weekday hourly analysis...")
    
    # Prepare data with weekend/weekday classification
    data_weekend = data.copy()
    data_weekend['is_weekend'] = data_weekend['datetime'].dt.dayofweek.isin([5, 6])
    data_weekend['day_type'] = data_weekend['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
    data_weekend['hour'] = data_weekend['datetime'].dt.hour
    
    # Get unique sites and models
    sites = sorted(data['site'].unique())
    models = sorted(data['model'].unique())
    
    # Create separate analysis for each metric (label)
    for metric_idx, (metric, metric_name, base_color) in enumerate(zip(METRICS, METRIC_NAMES, METRIC_COLORS)):
        print(f"Processing {metric_name}...")
        
        # Create subplot grid: one row per site, one column per model
        n_sites = len(sites)
        n_models = len(models)
        
        fig, axes = plt.subplots(n_sites, n_models, figsize=(6*n_models, 5*n_sites))
        fig.suptitle(f'{metric_name} - Weekend vs Weekday Hourly Patterns by Site', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Handle single site or single model cases
        if n_sites == 1 and n_models == 1:
            axes = np.array([[axes]])
        elif n_sites == 1:
            axes = axes.reshape(1, -1)
        elif n_models == 1:
            axes = axes.reshape(-1, 1)
        
        # Colors for weekend/weekday
        colors = {'Weekend': '#FF6B6B', 'Weekday': '#4ECDC4'}
        
        for i, site in enumerate(sites):
            for j, model in enumerate(models):
                ax = axes[i, j]
                
                # Filter data for this site and model
                site_model_data = data_weekend[
                    (data_weekend['site'] == site) & 
                    (data_weekend['model'] == model)
                ]
                
                if len(site_model_data) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{site} - {model}', fontweight='bold')
                    continue
                
                # Group by day type and hour
                hourly_patterns = site_model_data.groupby(['day_type', 'hour'])[metric].mean().reset_index()
                
                # Plot weekend and weekday patterns
                for day_type in ['Weekend', 'Weekday']:
                    day_data = hourly_patterns[hourly_patterns['day_type'] == day_type]
                    
                    if len(day_data) > 0:
                        ax.plot(day_data['hour'], day_data[metric], 
                               marker='o', label=day_type, linewidth=2.5,
                               color=colors[day_type], markersize=6,
                               linestyle='-' if day_type == 'Weekday' else '--')
                
                # Customize the plot
                ax.set_title(f'{site} - {model}', fontweight='bold', fontsize=12)
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel(f'Mean {metric_name}')
                ax.set_xticks(range(0, 24, 2))
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                
                # Add activity period backgrounds
                ax.axvspan(5, 8, alpha=0.1, color='gold', label='Morning Chorus' if i == 0 and j == 0 else "")
                ax.axvspan(17, 19, alpha=0.1, color='orange', label='Evening Chorus' if i == 0 and j == 0 else "")
                ax.axvspan(12, 15, alpha=0.1, color='lightgray', label='Midday Quiet' if i == 0 and j == 0 else "")
                
                # Set consistent y-axis limits for better comparison
                all_values = hourly_patterns[metric].values
                if len(all_values) > 0:
                    y_min, y_max = all_values.min(), all_values.max()
                    y_range = y_max - y_min
                    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        plt.tight_layout()
        analyzer.save_plot(fig, f'15_weekend_weekday_hourly_{metric}.png')
        
        # Create summary comparison plot for this metric
        fig_summary, ax_summary = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate overall weekend vs weekday patterns across all sites
        overall_patterns = data_weekend.groupby(['day_type', 'hour'])[metric].mean().reset_index()
        
        for day_type in ['Weekend', 'Weekday']:
            day_data = overall_patterns[overall_patterns['day_type'] == day_type]
            ax_summary.plot(day_data['hour'], day_data[metric], 
                           marker='o', label=day_type, linewidth=3,
                           color=colors[day_type], markersize=8,
                           linestyle='-' if day_type == 'Weekday' else '--')
        
        ax_summary.set_title(f'{metric_name} - Overall Weekend vs Weekday Pattern', 
                           fontweight='bold', fontsize=14)
        ax_summary.set_xlabel('Hour of Day')
        ax_summary.set_ylabel(f'Mean {metric_name}')
        ax_summary.set_xticks(range(0, 24, 2))
        ax_summary.grid(True, alpha=0.3)
        ax_summary.legend(loc='upper right', fontsize=12)
        
        # Add activity period backgrounds
        ax_summary.axvspan(5, 8, alpha=0.1, color='gold', label='Morning Chorus')
        ax_summary.axvspan(17, 19, alpha=0.1, color='orange', label='Evening Chorus')
        ax_summary.axvspan(12, 15, alpha=0.1, color='lightgray', label='Midday Quiet')
        
        plt.tight_layout()
        analyzer.save_plot(fig_summary, f'16_weekend_weekday_summary_{metric}.png')
generate_weekend_weekday_hourly_analysis(data, analyzer)