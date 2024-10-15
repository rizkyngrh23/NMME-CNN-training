# Taylor Diagram

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load and average observation data
def load_and_average_obs(files, variable_name, lat_range, lon_range):
    datasets = [xr.open_dataset(file, decode_times=False).sel(Y=lat_range, X=lon_range) for file in files]
    combined_data = xr.concat(datasets, dim='T')
    averaged_data = combined_data[variable_name].mean(dim='T').values
    return averaged_data

# Load and average model data
def load_and_average_model(file, variable_name, lat_range, lon_range):
    dataset = xr.open_dataset(file, decode_times=False).sel(Y=lat_range, X=lon_range)
    averaged_data = dataset[variable_name].mean(dim=set(dataset.dims) - {'Y', 'X'}).values
    return averaged_data

# Taylor diagram plotting function
def taylor_diagram(stddevs, correlations, labels, ref_stddev, colors, markers, fig):
    # Create polar plot for the Taylor diagram
    ax = fig.add_subplot(111, polar=True)
    
    # Set axis limits
    ax.set_ylim([0, max(stddevs + [ref_stddev]) * 1.5])
    
    # Set the angle direction and offset for a standard Taylor diagram
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    
    # Plot the reference observation point
    ax.plot([0], [ref_stddev], 'ro', label='Observed')
    
    # Add the dashed line representing the observation standard deviation (black color)
    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(theta, np.ones_like(theta) * ref_stddev, 'k--')

    # Plot model points with different colors and markers
    for stddev, corr, label, color, marker in zip(stddevs, correlations, labels, colors, markers):
        theta = np.arccos(corr)  # Correlation to theta conversion
        ax.plot(theta, stddev, marker, label=label, color=color)
    
    # Set the plot to represent a quarter-circle
    ax.set_xlim([0, np.pi / 2])
    
    # Add correlation labels at the appropriate angles
    correlations_labels = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    theta_labels = np.arccos(correlations_labels)
    ax.set_thetagrids(angles=np.degrees(theta_labels), labels=correlations_labels)

    # Add "Correlation" label on the curve
    ax.text(np.pi / 4, ax.get_ylim()[1] * 0.9, 'Correlation', rotation=-45, 
            horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    
    # Set labels and title
    ax.set_ylabel('Standard Deviation')
    
    # Add legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
    return ax

# Define months, models, and corresponding labels
months = ['nov', 'dec', 'jan', 'feb', 'mar']
models = ['canv2', 'cancm', 'IC3', 'nemo']
model_labels = ['CanSIPSv2', 'CANCM4i', 'CanSIPS-IC3', 'GEM-NEMO']

# Generate model files list
model_files = [
    (f'D:/{month}/predictions_{model}.nc', f'{label} issued {month}')
    for month in months
    for model, label in zip(models, model_labels)
]

obs_files = [
    'D:/',
    'D:/',
    'D:/'
]

lat_range = slice(-9, -5)
lon_range = slice(105, 115)

# Load observation data
obs_values = load_and_average_obs(obs_files, '__xarray_dataarray_variable__', lat_range, lon_range)
obs_values = obs_values.flatten()  # Flatten to 1D array for comparison

# Prepare arrays for model statistics
std_devs = []
correlations = []
model_labels_col = []
colors = []
markers = []

# Define distinct colors and markers
colors_list = cm.tab20(np.linspace(0, 1, len(models)))  # Different colors for each model
markers_list = ['o', 's', '^', 'D', 'v']  # Different markers for each month

# Process each model file for each month
for month in months:
    for model, color in zip(models, colors_list):
        # File path and label for the current month and model
        file = f'D:/{month}/predictions_{model}.nc'
        model_index = models.index(model)
        month_index = months.index(month)
        label = f'{model_labels[model_index]} issued {month}'
        marker = markers_list[month_index]  # Get the marker for the month
        
        try:
            model_values = load_and_average_model(file, 'predictions', lat_range, lon_range)
            model_values = model_values.flatten()  # Flatten to 1D array for comparison
            
            # Calculate statistics for Taylor diagram
            std_dev = np.std(model_values)
            corr = np.corrcoef(model_values, obs_values)[0, 1]
            
            std_devs.append(std_dev)
            correlations.append(corr)
            model_labels_col.append(label)
            colors.append(color)
            markers.append(marker)
        
        except FileNotFoundError:
            pass

# Reference standard deviation (observations)
ref_std = np.std(obs_values)

# Plot the Taylor diagram
fig = plt.figure(figsize=(12, 8))
ax = taylor_diagram(std_devs, correlations, model_labels_col, ref_std, colors, markers, fig)

plt.title('Taylor Diagram', size=14)
plt.tight_layout()  # Adjust layout to ensure nothing is clipped
plt.show()
