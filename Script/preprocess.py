# Extract Issued
# This script defines a function to process NetCDF datasets for multiple climate models 
# (CanSIPS-IC3, CanCM4i, CanSIPSv2, and GEM-NEMO) by extracting data for specified months 
# (November to March). It reads each model's dataset, selects data based on the month index, 
# extracts grid values, and saves the results as new NetCDF files in the output directory.

import xarray as xr
import os

def extract_issued_data(base_input_path, base_output_path):
    """
    Extracts monthly issued data from NetCDF files for specified climate models 
    and saves the results as new NetCDF files.

    Parameters:
    - base_input_path: str, the directory containing the input NetCDF files
    - base_output_path: str, the directory where the output NetCDF files will be saved
    """
    
    # List of models to process
    models = ['CanSIPS-IC3', 'CanCM4i', 'CanSIPSv2', 'GEM-NEMO']

    # Dictionary of issued months and their corresponding indices
    issued_months_and_indices = {
        'nov': 10,  # November
        'dec': 11,  # December
        'jan': 12,  # January  
        'feb': 13,  # February   
        'mar': 14   # March    
    }

    # Loop through each model
    for model in models:
        # Loop to process each month
        for issued, index_num in issued_months_and_indices.items():
            # Read the dataset for the current model
            input_file_path = os.path.join(base_input_path, f'{issued}_{model}.nc')
            
            # Check if the file exists to avoid errors
            if os.path.exists(input_file_path):
                ds = xr.open_dataset(input_file_path, decode_times=False)
            
                # Select data based on the month index
                ds_selected = ds.isel(S=index_num)
            
                # Extract grid values
                grid_values = ds_selected.to_dataframe().reset_index()
            
                # Save the results as a NetCDF file
                output_file_path = os.path.join(base_output_path, f'{issued}_{model}.nc')
                ds_selected.to_netcdf(output_file_path)
            
                print(f"File for {issued} from {model} has been successfully saved as {output_file_path}")
            
                # Display all grid values (be cautious if the dataset is large)
                print(f"All grid values for {issued} from {model}:")
                print(grid_values)
            else:
                print(f"Input file does not exist: {input_file_path}")

# Example usage
base_input_path = 'D:/Data/'
base_output_path = 'D:/Data/'
extract_issued_data(base_input_path, base_output_path)

####################################################################################################################################################
####################################################################################################################################################

# Accumulate per Lead
# This script defines a function to process NetCDF files for various climate models, 
# accumulating values based on lead time. It reads each model's dataset, sums the 
# values across the lead time dimension, and saves the results as new NetCDF files.

import xarray as xr
import os

def accumulate_per_lead(model_files, output_directory):
    """
    Accumulates values from NetCDF files based on lead time and saves the results as new NetCDF files.

    Parameters:
    - model_files: list of str, paths to the input NetCDF files for various climate models
    - output_directory: str, the directory where the output NetCDF files will be saved
    """
    
    # Loop to process each model file
    for file in model_files:
        # Read the dataset
        dataset = xr.open_dataset(file, decode_times=False)
        
        # Accumulate values based on lead time
        # Assuming the lead time dimension is the first dimension
        accumulated = dataset.groupby('L').sum()
        
        # Determine output file name based on input file name
        model_name = os.path.basename(file).replace('.nc', '')
        output_file = os.path.join(output_directory, f'sum_{model_name}.nc')
        
        # Save the accumulated results to a new NetCDF file
        accumulated.to_netcdf(output_file)
        
        # Display some grid values for checking
        # Get the first variable name from the dataset
        var_name = list(accumulated.data_vars.keys())[0]
        grid_values = accumulated[var_name].values
        
        # Display the variable name
        print(f"Variable name: {var_name}")
        
        # Display all grid values
        print(f"All grid values for {model_name}:")
        print(grid_values)
        
        print(f'File {file} has been processed and saved as {output_file}')

# Example usage
model_files = [
    'D:/Data/CanSIPSv2/'
    'D:/Data/CanCM4i/',
    'D:/Data/CanSIPS-IC3/'
    'D:/Data/GEM-NEMO/'
]

output_directory = 'D:/Output/'
accumulate_per_lead(model_files, output_directory)

####################################################################################################################################################
####################################################################################################################################################

# Filling missing data with IDW interpolation
# This script defines a function to perform Inverse Distance Weighting (IDW) interpolation 
# to fill in missing precipitation data in a NetCDF dataset. It reads the dataset, 
# applies IDW interpolation for each time step, and saves the resulting interpolated data 
# as a new NetCDF file.

import xarray as xr
import numpy as np
from scipy.spatial import cKDTree

def idw_interpolation(x, y, values, xi, yi, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation.

    Parameters:
    - x: array-like, known x-coordinates
    - y: array-like, known y-coordinates
    - values: array-like, known values at the known coordinates
    - xi: array-like, x-coordinates for interpolation
    - yi: array-like, y-coordinates for interpolation
    - power: float, the power parameter for weighting (default is 2)

    Returns:
    - interpolated_values: array-like, interpolated values at the given xi, yi coordinates
    """
    # Create a KDTree for efficient spatial queries
    tree = cKDTree(np.vstack((x, y)).T)
    distances, indices = tree.query(np.vstack((xi, yi)).T, k=len(x))
    
    # Initialize an array for interpolated values
    interpolated_values = np.zeros_like(xi)
    
    # Perform IDW interpolation
    for i in range(len(xi)):
        dist = distances[i]
        vals = values[indices[i]]
        # Avoid division by zero by replacing 0 with a small number
        dist = np.where(dist == 0, 1e-10, dis)
        weights = 1 / (dist ** power)
        interpolated_values[i] = np.sum(weights * vals) / np.sum(weights)
    
    return interpolated_values

# File path to the input NetCDF dataset
file_path = 'D:/Data/chirps.nc'
# Open the dataset
dataset = xr.open_dataset(file_path, decode_times=False)
data = dataset['precipitation']
print("Number of NaN values before interpolation:", data.isnull().sum().item())

# Extract coordinates and time
X = data['X'].values
Y = data['Y'].values
T = data['T'].values

# Initialize an array to hold the interpolated values
values = data.values
interpolated_values = np.full_like(values, np.nan)

# Loop through each time step to perform IDW interpolation
for t in range(values.shape[0]):
    # Create a grid of x and y coordinates
    grid_x, grid_y = np.meshgrid(X, Y)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    values_at_t = values[t].ravel()
    
    # Identify known points and values (non-NaN)
    mask = ~np.isnan(values_at_t)
    known_points = points[mask]
    known_values = values_at_t[mask]
    
    # Perform IDW interpolation for the current time step
    interpolated_at_t = idw_interpolation(
        known_points[:, 0], known_points[:, 1],
        known_values,
        points[:, 0], points[:, 1]
    )
    
    # Store the interpolated values in the appropriate time slice
    interpolated_values[t] = interpolated_at_t.reshape(len(Y), len(X))

# Create a new xarray DataArray for the interpolated data
interpolated_data = xr.DataArray(interpolated_values, coords=[T, Y, X], dims=['T', 'Y', 'X'])
print("Number of NaN values after IDW interpolation:", interpolated_data.isnull().sum().item())

# Save the interpolated data to a new NetCDF file
interpolated_data.to_netcdf('D:/Data/.nc')
print("IDW interpolation completed, and data saved to chirps_fix.nc")

####################################################################################################################################################
####################################################################################################################################################

# Regrid with IDW interpolation
# This script performs Inverse Distance Weighting (IDW) interpolation on climate model data
# to downscale the spatial resolution. It processes multiple months and models, saving the 
# interpolated data as new NetCDF files.

import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import os

# List of months to be processed
months = ['nov', 'dec', 'jan', 'feb', 'mar']

# List of model names
model_names = ['canv2', 'cancm', 'IC3', 'nemo']

def idw_interpolation(lons, lats, values, lon_new, lat_new, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation to regrid data.

    Parameters:
    - lons: array-like, original longitude coordinates
    - lats: array-like, original latitude coordinates
    - values: array-like, values corresponding to the original coordinates
    - lon_new: array-like, new longitude grid for interpolation
    - lat_new: array-like, new latitude grid for interpolation
    - power: float, the power parameter for weighting (default is 2)

    Returns:
    - interpolated_values: array-like, values interpolated to the new grid
    """
    # Create a meshgrid for the new latitude and longitude coordinates
    grid_lat, grid_lon = np.meshgrid(lat_new, lon_new, indexing='ij')
    grid_points = np.vstack((grid_lat.flatten(), grid_lon.flatten())).T
    
    # Flatten the input coordinates and values
    points = np.vstack((lats.flatten(), lons.flatten())).T
    values = values.flatten()
    
    # Create a KDTree for efficient distance computation
    tree = cKDTree(points)
    
    # Perform IDW interpolation
    distances, indices = tree.query(grid_points, k=len(points), p=2)  # k is the number of nearest neighbors
    distances = np.maximum(distances, 1e-10)  # Avoid zero distances
    weights = 1 / distances**power
    weighted_values = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
    
    return weighted_values.reshape(len(lat_new), len(lon_new))

# Loop to process each month and model
for month in months:
    for model in model_names:
        # Define the input file path
        file = f'D:/Kuliah/Skripsi/Data_baru/{month}/{month}_{model}.nc'
        
        # Read the dataset with decode_times=False
        dataset = xr.open_dataset(file, decode_times=False)
        
        # Extract the first variable
        var_name = list(dataset.data_vars.keys())[0]
        data = dataset[var_name]
        
        # Get original coordinate information
        lons = dataset.X.values  # Longitude
        lats = dataset.Y.values  # Latitude
        leads = dataset.L.values  # Lead times
        values = data.values

        # Determine the boundaries of the original coordinates
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        # Define new grid resolution
        lon_new = np.arange(lon_min, lon_max + 0.05, 0.05)  # New longitude grid
        lat_new = np.arange(lat_min, lat_max + 0.05, 0.05)  # New latitude grid
        
        # Create a meshgrid for the original coordinates
        lons_grid, lats_grid = np.meshgrid(lons, lats)
        
        # Initialize array to hold the interpolated values for each lead time
        sst_new = np.empty((len(leads), len(lat_new), len(lon_new)))
        sst_new.fill(np.nan)

        # Loop through each lead time
        for i, lead in enumerate(leads):
            # Extract data for the specific lead time
            values_lead = values[i, :, :]

            # Use IDW for interpolation
            grid_z = idw_interpolation(lons_grid, lats_grid, values_lead, lon_new, lat_new)
            sst_new[i, :, :] = grid_z

            # Check for NaN values in the interpolated data
            if np.isnan(grid_z).any():
                print(f'Warning: NaN values found in the interpolated data for lead time {lead} in file {file}')

        # Create a new dataset with the downscaled resolution
        ds_new = xr.Dataset(
            {
                var_name: (['L', 'Y', 'X'], sst_new)  # Adjusting to the new variable
            },
            coords={
                'L': leads,
                'Y': lat_new,
                'X': lon_new
            }
        )
        
        # Define the output file name based on the input file name
        model_name = os.path.basename(file).replace('.nc', '')
        output_file = f'D:/{month}/ds_{model_name}.nc'
        
        # Save the interpolated results to a new NetCDF file
        ds_new.to_netcdf(output_file)
        
        print(f'Downscaling for {file} has been processed and saved as {output_file}')
