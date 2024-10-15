# NMME Data Training with CNN for SPI Prediction

This project uses Convolutional Neural Networks (CNN) to evaluate various NMME (North American Multi-Model Ensemble) models for predicting the Standardized Precipitation Index (SPI) based on 30 years of historical precipitation data from 1981-2010. The models being evaluated are:

- **CanSIPSv2**
- **CanCM4i**
- **CanSIPS-IC3**
- **GEM-NEMO**

The aim of this project is to determine which model performs best in predicting SPI using only precipitation data. Evaluation is done using a **Taylor diagram** and **spatio-temporal visualizations** of RMSE, standard deviation, and correlation.

## Data

- **Time Period**: 1981-2010 (30 years)
- **Data Used**: Historical monthly precipitation data from NMME models (CanSIPSv2, CanCM4i, CanSIPS-IC3, GEM-NEMO).
- **Target**: Standardized Precipitation Index (SPI) calculated from precipitation data.

## Objective

To evaluate the performance of multiple climate models for SPI prediction, focusing on:

- **Root Mean Square Error (RMSE)**
- **Standard Deviation (stdev)**
- **Correlation Coefficient**
- **Skill Score**

## Methodology

1. **Data Preprocessing**:
   - Precipitation data from each model is preprocessed and normalized.
   - SPI values are calculated for each model based on historical precipitation data.
   
2. **Model Training**:
   - A CNN is trained on the precipitation data to predict SPI for each model.
   - The model uses spatial-temporal features from the precipitation data for prediction.

3. **Evaluation**:
   - A **Taylor diagram** is used to compare the performance of each model in terms of correlation, RMSE, and standard deviation.
   - **Spatio-temporal visualizations** are generated for the RMSE, standard deviation, and correlation for a detailed comparison.

## Tools & Libraries

- **Python**: Used for data processing and model training.
- **TensorFlow/Keras**: Deep learning library for building and training the CNN model.
- **Numpy & Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: For visualizations including Taylor diagrams and spatio-temporal plots.
- **Basemap**: For spatial visualizations.
- **NetCDF4**: Handling of NMME model data in NetCDF format.

## Results
- **Taylor Diagrams**: Compare model performance in terms of RMSE, standard deviation, and correlation.
- **Spatio-Temporal Visualizations**: RMSE, standard deviation, correlation, and skill score plotted over time and space to assess model accuracy and behavior across different regions and periods.

1. **Taylor Diagram**
   
   <div align="center">
      <img src="https://github.com/user-attachments/assets/037a6ef7-5942-4309-9faf-b5ffad560c62" alt="RMSE" width="800" />
   </div>


3. **RMSE**

   <div align="center">
      <img src="https://github.com/user-attachments/assets/5402fffd-72c3-4164-92e7-4a24b32a7757" alt="RMSE" width="800" />
   </div>
   
4. **Standard Deviation**

   <div align="center">
      <img src="https://github.com/user-attachments/assets/6faba311-14b2-4289-b2e8-9d00fbe6e163" alt="RMSE" width="800" />
   </div>

5. **Spatial Correlation**

    <div align="center">
      <img src="https://github.com/user-attachments/assets/d3db7ad9-071f-4786-a121-c471b39021b9" alt="RMSE" width="800" />
   </div>
    
6. **Skill Score**

   <div align="center">
      <img src="https://github.com/user-attachments/assets/bc1d58fc-bbd1-4490-a55f-f26dc51b1d0a" alt="RMSE" width="800" />
   </div>


## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rizkyngrh23/NMME-CNN-training.git
