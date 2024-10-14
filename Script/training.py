import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress, pearsonr
import keras as ks
import tensorflow as tf
import os
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Function to calculate R-squared
def r_squared(pred, obs):
    slope, intercept, r_value, _, _ = linregress(pred.flatten(), obs.flatten())
    return r_value ** 2

# Data directories
model_files = [
    ('D:/Kuliah/Skripsi/Data_baru/nov/predictions_canv2.nc', 'CanSIPSv2'),
    ('D:/Kuliah/Skripsi/Data_baru/nov/predictions_cancm.nc', 'CANCM4i'),
    ('D:/Kuliah/Skripsi/Data_baru/nov/predictions_IC3.nc', 'CanSIPS-IC3'),
    ('D:/Kuliah/Skripsi/Data_baru/nov/predictions_nemo.nc', 'GEM-NEMO')
]

obs_files = [
    'D:/Kuliah/Skripsi/Data_baru/spi_mei_obs.nc',
    'D:/Kuliah/Skripsi/Data_baru/spi_jun_obs.nc',
    'D:/Kuliah/Skripsi/Data_baru/spi_jul_obs.nc'
]

lat_range = slice(-9, -5)
lon_range = slice(105, 115)

lead_times = (6.5, 8.5)

# Model parameters
num_epochs = 50
learning_rate = 0.0001
batch_size = 1

# Load and average observation data
def load_and_average_obs(files, variable_name):
    try:
        datasets = [xr.open_dataset(file, decode_times=False).sel(Y=lat_range, X=lon_range) for file in files]
        combined_data = xr.concat(datasets, dim='T')
        averaged_data = combined_data[variable_name].mean(dim='T').values
        return averaged_data
    except Exception as e:
        print(f"Error loading observation files: {e}")
        return None

# Load observation data
obs_values = load_and_average_obs(obs_files, '__xarray_dataarray_variable__')
if obs_values is None:
    raise ValueError("Failed to load observation data.")

# Reshape observation data for CNN
obs_values = np.expand_dims(obs_values, axis=-1)  # Add channel dimension
obs_values = np.expand_dims(obs_values, axis=0)  # Add batch dimension

# Function to load and average model data
def load_and_average_model(file, variable_name):
    try:
        dataset = xr.open_dataset(file, decode_times=False).sel(Y=lat_range, X=lon_range)
        averaged_data = dataset[variable_name].sel(L=slice(*lead_times)).mean(dim='L').values
        return averaged_data, dataset
    except Exception as e:
        print(f"Error loading model file {file}: {e}")
        return None, None

# Dictionary to store performance metrics for each model
performance_summary = []
all_losses = []  # List to store loss values for learning curve

# Function for CNN model definition
def create_cnn_model(input_shape):
    cnn = ks.models.Sequential([
        ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        ks.layers.MaxPooling2D(pool_size=(2, 2)),
        ks.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        ks.layers.MaxPooling2D(pool_size=(2, 2)),
        ks.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        ks.layers.Flatten(),
        ks.layers.Dense(64, activation='relu'),
        ks.layers.Dense(np.prod(obs_values.shape[1:]), activation='linear'),
        ks.layers.Reshape(obs_values.shape[1:])
    ])
    return cnn

# Loop over each model
for model_file, model_name in model_files:
    # Load and average model data
    model_values, dataset = load_and_average_model(model_file, 'prec')
    if model_values is None or dataset is None:
        continue  # Skip this model if loading fails

    # Debug: Print the shape of the data
    print(f'Loaded model data shape for {model_name}: {model_values.shape}')
    print(f'Loaded obs data shape: {obs_values.shape}')

    # Min-max scale the model data
    min_model_value = model_values.min()
    max_model_value = model_values.max()
    model_values_scaled = (model_values - min_model_value) / (max_model_value - min_model_value)

    # Min-max scale the observation data using the same range as the model data
    min_obs_value = obs_values.min()
    max_obs_value = obs_values.max()
    obs_values_scaled = (obs_values - min_obs_value) / (max_obs_value - min_obs_value)

    # Reshape model data for CNN
    model_values_scaled = np.expand_dims(model_values_scaled, axis=-1)  # Add channel dimension
    model_values_scaled = np.expand_dims(model_values_scaled, axis=0)  # Add batch dimension

    # Ensure model_values_scaled and obs_values_scaled have the same shape
    if model_values_scaled.shape[1:] != obs_values_scaled.shape[1:]:
        raise ValueError(f'Model and observation data shapes do not match: {model_values_scaled.shape} vs {obs_values_scaled.shape}')

    # Create CNN model
    cnn = create_cnn_model(model_values_scaled.shape[1:])

    # Compile model
    cnn.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    # Split data into training and validation sets
    # Since we have only one data point per model, we'll use the same data for validation.
    model_train, model_val = model_values_scaled, model_values_scaled
    obs_train, obs_val = obs_values_scaled, obs_values_scaled

    # Training loop
    history = cnn.fit(model_train, obs_train, 
                      validation_data=(model_val, obs_val),
                      epochs=num_epochs, 
                      batch_size=batch_size, 
                      verbose=1)

    # Store loss values for learning curve
    all_losses.append(history.history['loss'])

    # Plotting the learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='b')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='r')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Predict and calculate metrics
    outputs = cnn.predict(model_values_scaled)
    mae = mean_absolute_error(obs_values_scaled.flatten(), outputs.flatten())
    rmse = np.sqrt(mean_squared_error(obs_values_scaled.flatten(), outputs.flatten()))
    r2 = r_squared(outputs, obs_values_scaled)

    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(obs_values_scaled.flatten(), outputs.flatten())

    # Store performance metrics in the summary
    performance_summary.append({
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2,
        'Pearson Correlation': pearson_corr
    })

    # Flatten the outputs and observations for plotting
    obs_flat = obs_values_scaled.flatten()
    preds_flat = outputs.flatten()

    # Plot a sample of the data
    sample_size = 100  # Adjust this to change the size of the sample being plotted
    sample_indices = np.arange(sample_size)

    # Line Plot for Sample
    plt.figure(figsize=(10, 5))
    plt.plot(sample_indices, obs_flat[:sample_size], label='Observed', color='b')
    plt.plot(sample_indices, preds_flat[:sample_size], label='Predicted', color='r', linestyle='--')
    plt.title(f'Line Plot (Sample) - {model_name}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save CNN result to new NetCDF file
    output_dataset = xr.Dataset(
        {
            'predictions': (['Y', 'X'], outputs.squeeze())
        },
        coords={
            'Y': dataset['Y'].values,
            'X': dataset['X'].values
        }
    )
    output_file_path = f'D:/Kuliah/Skripsi/Data_baru/mar/predictions_{model_name}.nc'
    output_dataset.to_netcdf(output_file_path)
    print(f'Saved predictions to {output_file_path}')

# Plot average learning curve across all models
average_losses = np.mean(np.array(all_losses), axis=0)
plt.figure(figsize=(10, 5))
plt.plot(average_losses, label='Average Training Loss', color='g')
plt.title('Average Learning Curve Across All Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print performance summary
print("\nPerformance Summary:")
print(f"{'Model':<15}{'MAE':<10}{'RMSE':<10}{'R-squared':<15}{'Pearson Correlation':<20}")
for summary in performance_summary:
    print(f"{summary['Model']:<15}{summary['MAE']:<10.4f}{summary['RMSE']:<10.4f}{summary['R-squared']:<15.4f}{summary['Pearson Correlation']:<20.4f}")
