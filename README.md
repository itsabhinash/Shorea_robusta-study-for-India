# Bamboo Study for India

This project uses Google Earth Engine (GEE) and geospatial data analysis tools to model Shorea robusta distribution by using spectral and backscatter parameters for India. The workflow includes importing, processing, and exporting geospatial datasets to facilitate Shorea robusta-related studies. Additionally, it incorporates a machine learning model for predictive analysis using Random Forest.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Data Processing](#data-processing)
  - [Tree Presence](#tree-presence)
  - [Spectral Data Extraction](#spectral-data-extraction)
  - [Backscatter Data Extraction](#backscatter-data-extraction)
- [Machine Learning Model](#machine-learning-model)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Raster Prediction](#raster-prediction)
- [Exporting Data](#exporting-data)

---

## Project Overview

This repository provides scripts for geospatial analysis focused on:
- Identifying tree presence areas.
- Extracting spectral data from Sentinel-2 imagery.
- Analyzing backscatter data from Sentinel-1 imagery.
- Predicting spatial data patterns using a Random Forest model.

## Features

- Tree presence masking using ESA WorldCover 10m data.
- Time-period-specific spectral and backscatter data extraction.
- Predictive analysis using Random Forest for raster data.

## Dependencies

Ensure the following Python libraries are installed:
- `earthengine-api`
- `geemap`
- `ipyleaflet`
- `matplotlib`
- `numpy`
- `pandas`
- `rasterio`
- `scikit-learn`
- `tqdm`

Install dependencies using:
```bash
pip install geemap ipyleaflet matplotlib numpy pandas rasterio scikit-learn tqdm
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Authenticate and initialize Google Earth Engine (GEE):
   ```python
   import ee
   ee.Authenticate()
   ee.Initialize()
   ```

3. Update the `India_boundary` path to your study area's feature collection:
   ```python
   India_boundary = ee.FeatureCollection('path/India_boundary')
   ```

## Usage

Run the scripts to process geospatial data as per your requirements:
```bash
python <script_name>.py
```

Ensure you modify input parameters (time periods, regions, etc.) in the scripts to match your study area.

---

## Data Processing

### Tree Presence
- Data Source: ESA WorldCover 10m.
- Unwanted land cover classes masked to isolate tree areas.

### Spectral Data Extraction
- Data Source: Sentinel-2 Harmonized (`COPERNICUS/S2_SR_HARMONIZED`).
- Filters cloud, shadow, and cirrus effects.
- Processes spectral data for specified time periods.

### Backscatter Data Extraction
- Data Source: Sentinel-1 (`COPERNICUS/S1_GRD`).
- Extracts VV and VH polarizations for descending orbits.

---

## Machine Learning Model

### Model Training and Evaluation

- The Random Forest model is used for predictive analysis of raster data.
- **Data Preparation**:
  - Raster values are extracted at point locations using a chunked approach to handle large datasets.
  - Features are normalized using Min-Max scaling.
  - The training data includes spatial and raster-derived features.

- **Training**:
  - Hyperparameter tuning is performed using GridSearchCV.
  - A progress bar (`tqdm`) is used for real-time feedback during the process.

- **Evaluation**:
  - Metrics such as overall accuracy, confusion matrix, and class-wise accuracy are computed.
  - The best model and its parameters are logged.

### Raster Prediction

- The trained model predicts new raster data in chunks.
- Progress bars track prediction status.
- The output raster is saved with metadata and no-data values handled appropriately.

---

## Exporting Data

Export processed data to Google Drive using `geemap.ee_export_image_to_drive`:
```python
geemap.ee_export_image_to_drive(
   image=processed_image,
   description="Export_Description",
   folder="Export_Folder",
   scale=10,
   region=India_boundary.geometry(),
   maxPixels=1e13
)
```

---
