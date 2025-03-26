# Shorea robusta study for india

# Geemap

import ee
import geemap
import json
import os
import requests
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import joblib
import glob
import os
from tqdm import tqdm
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

ee.Initialize()

# Load the India Boundary (Study area)
India_boundary = ee.FeatureCollection('path/India_boundary')


# Tree presence
# ESA worldcover 10m
esa_lc = ee.ImageCollection('ESA/WorldCover/v200').first()

# Define the land cover classes to be masked.
classes_to_mask = [20,30,40,50,60,70,80,90,95,100]

# Create a mask where the undesired land cover classes are assigned a value of 0.
mask = esa_lc.neq(classes_to_mask[0])
for i in range(1, len(classes_to_mask)):
    mask = mask.And(esa_lc.neq(classes_to_mask[i]))

# Apply the mask to the land cover dataset to get tree presence areas.
masked_lc = esa_lc.mask(mask).clip(India_boundary)


# Processing for spectral data extraction
# Considered time periods
    #  Summer median = '2023-03-01 to '2023-05-31'
    #  Winter median = '2023-10-01 to '2023-12-31'
# Considered bands
    # B2 , B3 , B4 , B5 ,B6 B7 , B8 , B8a , B11 , B12

def maskCloudAndShadows(image):
    cloudProb = image.select('MSK_CLDPRB')
    snowProb = image.select('MSK_SNWPRB')
    cloud = cloudProb.lt(5)
    snow = snowProb.lt(5)
    scl = image.select('SCL')
    shadow = scl.eq(3)  # 3 = cloud shadow
    cirrus = scl.eq(10)  # 10 = cirrus
    # Cloud probability less than 5% or cloud shadow classification
    mask = (cloud.And(snow)).And(cirrus.neq(1)).And(shadow.neq(1))
    return image.updateMask(mask)

startDate = '2023-03-01'
endDate = '2023-05-31'

collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(startDate, endDate).map(maskCloudAndShadows).filterBounds(India_boundary)
collection_median = collection.select('B1').median().updateMask(masked_lc).divide(10000).rename('collection_median')

# Exporting band data to drive
geemap.ee_export_image_to_drive(
   collection_median.clip(India_boundary) , 
   description="B1_India_summer", 
   folder="gee",
   scale=10 , 
   region=India_boundary.geometry().dissolve() , 
   maxPixels=1e13
)


# Processing for backscatter data extraction
# Considered time periods
    #  Summer median = '2023-03-01 to '2023-05-31'
    #  Winter median = '2023-10-01 to '2023-12-31'
# Considered bands
    # VV,VH

# Import the Sentinel-1 ImageCollection
sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')

# Function to filter Sentinel-1 data based on orbit and polarization pair
def get_filtered_images(orbit_pass, polarization_pair):
    return sentinel1.filterBounds(India_boundary) \
                    .filterDate('2023-03-01', '2023-05-31') \
                    .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass)) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization_pair[0])) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization_pair[1])) \
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))

# Filter for VV + VH polarizations
vv_vh_desc = get_filtered_images('DESCENDING', ['VV', 'VH'])

# Create median composites for each combination
combined = vv_vh_desc.select('VV', 'VH').median().clip(India_boundary).rename(['VV_desc', 'VH_desc']).updateMask(masked_lc)

# Exporting band data to drive
geemap.ee_export_image_to_drive(
    combined.clip(India_boundary),
    description="backscatter_summer",
    folder="gee",
    scale=10,
    region=India_boundary.geometry(),  
    maxPixels=1e13  
)


# Model

shapefile_path = "C:/file/path.shp"
raster_folder = r"file/path/"
csv_output_path = "C:/file/path.csv"
model_output_path = "C:/file/file.pkl"
predicted_tif_path = "C:/file/file.tif"

# Get all .tif raster files from the folder
raster_paths = sorted(glob.glob(os.path.join(raster_folder, "*.tif")))
print(f"Rasters Found: {len(raster_paths)} files")

# Read the shapefile
points = gpd.read_file(shapefile_path)

# -------------------------
# Function: Raster Extraction (Parallel Processing)
# -------------------------
def extract_parallel(raster_path, points):
    """Extracts raster values for all points from a single raster file efficiently."""
    with rasterio.open(raster_path) as src:
        coords = [(point.x, point.y) for point in points.geometry]
        values = list(tqdm(src.sample(coords), total=len(coords), desc=f"Processing {os.path.basename(raster_path)}"))
        nodata_value = src.nodata
        return [v[0] if v[0] != nodata_value else np.nan for v in values]

# Extract raster values efficiently
all_values = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(extract_parallel)(rp, points) for rp in tqdm(raster_paths, desc="Extracting Rasters")
)

# Convert list to NumPy array and filter NoData
all_values = np.array(all_values, dtype=np.float32).T  # Transpose for correct shape
valid_indices = ~np.isnan(all_values).any(axis=1)  # Keep only valid rows
features, valid_indices = all_values[valid_indices], np.where(valid_indices)[0]

# Create and Save DataFrame
filtered_classes = points['classvalue'].iloc[valid_indices].reset_index(drop=True)
additional_columns = points[['Lat', 'Long', 'classname']].iloc[valid_indices].reset_index(drop=True)

data = pd.DataFrame(features, columns=[f"Feature_{i}" for i in range(len(raster_paths))])
data['Class'] = filtered_classes.astype(np.int8)
data[['Lat', 'Long', 'Classname']] = additional_columns

# Save extracted features as CSV
data.to_csv(csv_output_path, index=False)
print(f"Training data saved to {csv_output_path}")

# -------------------------
# Train Random Forest Model
# -------------------------

# Prepare training data
X = data.drop(columns=["Class", "Lat", "Long", "Classname"]).to_numpy(dtype=np.float32)
y = data["Class"].to_numpy(dtype=np.int8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True
)

print("Training Random Forest Model...")
rf.fit(X_train, y_train)

# Save trained model
joblib.dump(rf, model_output_path)
print(f"Model saved to {model_output_path}")

# Model Evaluation
print(f"Training Accuracy: {rf.score(X_train, y_train):.2f}")
print(f"Validation Accuracy: {rf.score(X_test, y_test):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf.predict(X_test)))
print("Classification Report:")
print(classification_report(y_test, rf.predict(X_test)))

# Feature Importance
importances = rf.feature_importances_
feature_names = [f"Feature_{i}" for i in range(len(raster_paths))]

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:\n", importance_df)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()  # Most important feature at the top
plt.show()

# -------------------------
# 🔹 Predict Raster Data
# -------------------------

def predict_raster(input_raster_paths, model, output_raster_path, chunk_size=512):
    """Predicts raster data using a trained model in chunks."""
    rasters = [rasterio.open(path) for path in input_raster_paths]
    meta = rasters[0].meta.copy()
    width, height = meta['width'], meta['height']
    output_data = np.full((height, width), np.nan, dtype=np.float32)  # Use NaN for no-data

    with tqdm(total=height, desc="Predicting raster rows") as pbar:
        for row_start in range(0, height, chunk_size):
            row_end = min(row_start + chunk_size, height)
            chunk_height = row_end - row_start
            
            # Extract raster values for the chunk
            img_chunk = np.stack([
                np.where(r.read(1, window=Window(0, row_start, width, chunk_height)) == r.nodata,
                         np.nan, r.read(1, window=Window(0, row_start, width, chunk_height)))
                .astype(np.float32).reshape(chunk_height * width)
                for r in rasters
            ], axis=1)
            
            # Identify valid pixels (non-NaN)
            mask = np.isnan(img_chunk).any(axis=1)
            valid_pixels = img_chunk[~mask]
            
            # Predict valid pixels
            predictions = np.full(img_chunk.shape[0], np.nan, dtype=np.float32)
            if len(valid_pixels) > 0:
                predictions[~mask] = model.predict(valid_pixels)
            
            # Store predictions in output raster
            output_data[row_start:row_end, :] = predictions.reshape(chunk_height, width)
            pbar.update(chunk_height)
    
    # Save predicted raster
    meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(output_data, 1)

    # Close raster files
    for r in rasters:
        r.close()

# Run prediction
predict_raster(raster_paths, rf, predicted_tif_path)
print(f"Predicted raster saved to {predicted_tif_path}")














