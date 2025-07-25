# HM_Land Instance Segmentation using SAM Models

This project was developed for the **HM Land Registry Data Science Challenge**. It performs high-resolution instance segmentation and reference detection from scanned land registry maps using:

- **Segment Anything Models (SAM)** from Meta AI  
- **PaddleOCR**  
- **Geospatial tools** like GeoPandas and Shapely

The pipeline automates the detection of plot boundaries and their associated reference labels, exporting the final results as geospatial data.

---

## 🔧 Requirements

- Python 3.10  
- Anaconda (recommended for environment management)  
- NVIDIA GPU with CUDA 11.8 support (required for GPU acceleration with PyTorch and PaddlePaddle)

---

## ⚙️ Setup & Installation

### i. Clone the Repository

```bash
git clone https://github.com/Consolelives/HM-Land-Registry_DataScience-Challenge.git
cd HM_Land-Instance-Segmentation

```
### ii. Create and Activate a Conda Environment
conda create -n hm_land_seg python=3.10 -y
conda activate hm_land_seg

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
