# Map-Seg-Paddle-Linker

Automatic Extraction of Red Boundaries & Reference Numbers from Scanned Maps

This project uses advanced computer vision and OCR to detect red plot boundaries and read red reference numbers from scanned maps, then exports the results as fully usable geospatial data.

Ideal for:

  - Land registry digitization

  - Urban planning and cadastral mapping

  - Historical map preservation and analysis


#  Key Features
Red Line Segmentation – Detect and extract precise plot boundaries.

Red Number Recognition – Automatically read reference numbers using OCR.

Boundary-to-Reference Linking – Match each polygon to its corresponding label.

GIS-Compatible Output – Export as .gpkg or shapefile for QGIS/ArcGIS.

Modular Jupyter notebooks & scripts for full or step-by-step execution.

#  How It Works
Example Workflow:
Input – A scanned or photographed map with red plot boundaries and red reference numbers.

Preprocessing –

Convert to HSV color space

Isolate red pixels (lines & text)

Remove noise and skeletonize lines

Segmentation – Use Segment Anything Model (SAM) to detect plot polygons.

OCR Detection – Use PaddleOCR to read red reference numbers from the image.

Data Linking – Match each detected polygon to its corresponding reference number.

Export – Save as GeoPackage (.gpkg) or shapefile for GIS systems.



# Visual Overview

Input Map	   
<img width="2990" height="2770" alt="trim_class" src="https://github.com/user-attachments/assets/a7abbace-4d08-41d6-8cbe-039e0e8624b0" />

# Skeletonized Image

<img width="2990" height="2770" alt="output_image_4" src="https://github.com/user-attachments/assets/4485a03c-c2ad-4393-bef0-8c5b81e34b40" />

<img width="2990" height="2770" alt="skeleton_image" src="https://github.com/user-attachments/assets/336357be-0039-4163-806d-84268278eb28" />

## Extracted Red Lines & Numbers	
<img width="953" height="827" alt="segment_1" src="https://github.com/user-attachments/assets/202381e5-a161-4fc5-940d-ad658f49da4a" />

<img width="5980" height="2770" alt="trimmed_ocr_res_img" src="https://github.com/user-attachments/assets/970ea435-272c-4207-b53c-6e709c8c9f6b" />

<img width="136" height="89" alt="97_0554" src="https://github.com/user-attachments/assets/3165a478-2940-4e40-8716-1b341b58e060" />

# Final GIS Output


<img width="2131" height="1979" alt="Geopandas_map" src="https://github.com/user-attachments/assets/d50020a3-431d-419f-8ce6-4424f78728b7" />

---

# 🔧 Requirements

- Python 3.10  
- Anaconda (recommended for environment management)  
- NVIDIA GPU with CUDA 11.8 support (required for GPU acceleration with PyTorch and PaddlePaddle)

---

## ⚙️ Setup & Installation

### i. Clone the Repository

```bash
# Clone repository
git clone https://github.com/Consolelives/Map-Seg-Paddle-Linker.git
cd Map-Seg-Paddle-Linker



```
### Create environment
conda create -n map_seg python=3.10 -y
conda activate map_seg

```

```
### iii. Install all Depedencies(with CUDA 11.8)
pip install -r requirements.txt

```

📘 Notebooks Overview
This repository contains multiple Jupyter notebooks that walk through each step of the pipeline. Please open and run them in the order below for clarity:

```
1. IMAGE PROCESSING -CV2-SUBMITTED.ipynb
Purpose:
Image preprocessing for detection and segmentation.

Trimmed original images

Extracted red lines in HSV space

Applied skeletonization

Saved cleaned data to CSV
```

```
2. PADDLE OCR - SUBMITTED.ipynb
Purpose:
Text detection using PaddleOCR.

Detected reference text with confidence scores above 82%

Data cleaning performed for accuracy

Note: To install PaddleOCR, uncomment and run the relevant markdown cell in the notebook

Reference confidence scores shown in Cell 14
```


```
3. SAM MODELS - SUBMITTED2.ipynb
Purpose:
Instance segmentation using SAM variants.

Used SAM-B, SAM-L, and MobileSAM from Ultralytics

Detected shapes from skeletonized image

Achieved 87%+ confidence on most instances

Confidence scores are displayed in Cell 28 or under the markdown:

Using the Confidence Column, get the rows with the highest confidence

```

```
4. COMBINE THE IMAGE AND REFERENCE DATAFRAMES - Submitted.ipynb
Purpose:
Match detected shapes and reference text.

Combined results from PaddleOCR and SAM segmentation

Mapped text to shapes

Exported the final metadata into a .pkl file for use in GeoPandas

```

```
5. GEOPANDAS - SUBMITTED.ipynb
Purpose:
Visualize and export the spatial data.

Saved results as a GeoPackage (.gpkg)

Plotted segmented instances with overlaid references
```

```
✅ OR... Just run everything at once
Use Run all Process.ipynb to execute the full pipeline in one go.

No need to modify other notebooks — all logic is modularized and also available in standalone Python scripts.

```

```
🙌 Acknowledgements
Meta AI – Segment Anything

Ultralytics YOLOv8

PaddleOCR

PyTorch

GeoPandas
```
