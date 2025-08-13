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
1. Input – A scanned or photographed map with red plot boundaries and red reference numbers.
2. Preprocessing –

    - Convert to HSV color space
    - Isolate red pixels (lines & text)
    - Remove noise and skeletonize lines

3. Segmentation – Use Segment Anything Model (SAM) to detect plot polygons.
4. OCR Detection – Use PaddleOCR to read red reference numbers from the image.
5. Data Linking – Match each detected polygon to its corresponding reference number.
6. Export – Save as GeoPackage (.gpkg) or shapefile for GIS systems



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

🚀 Usage
You can run each step individually via notebooks, or execute the full pipeline at once.

Full pipeline:


jupyter notebook Run_all_Process.ipynb

Step-by-step notebooks:

IMAGE_PROCESSING.ipynb – Isolate red boundaries & numbers

PADDLE_OCR.ipynb – Detect and read reference numbers

SAM_MODELS.ipynb – Segment plots into polygons

COMBINE_DATA.ipynb – Link polygons with reference numbers

GEOPANDAS.ipynb – Export results as GIS-ready data


