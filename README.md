# Map-Seg-Paddle-Linker

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

From Map Image to GIS Data in Minutes — AI-Powered Boundary & Label Extraction  
Automatically detect red plot boundaries and reference numbers from scanned maps, ready for GIS.

Ideal for:

- Land registry digitization  
- Urban planning and cadastral mapping  
- Historical map preservation and analysis  

---

## Table of Contents

- [Key Features](#key-features)  
- [How It Works](#how-it-works)  
- [Visual Overview](#visual-overview)  
- [Requirements](#-requirements)  
- [Setup & Installation](#-setup--installation)  
- [Usage](#usage)  
- [Why This Matters](#why-this-matters)  
- [Contact](#contact)  
- [Acknowledgements](#acknowledgements)  

---

## Key Features

- **Red Line Segmentation** – Detect and extract precise plot boundaries.  
- **Red Number Recognition** – Automatically read reference numbers using OCR.  
- **Boundary-to-Reference Linking** – Match each polygon to its corresponding label.  
- **GIS-Compatible Output** – Export as `.gpkg` or shapefile for QGIS/ArcGIS.  
- **Modular Notebooks & Scripts** – Run step-by-step or execute full pipeline.  

---

## How It Works

📄 **Scanned Map** → 🎨 **Red Pixel Isolation** → 🖼️ **SAM Segmentation** → 🔢 **PaddleOCR** → 🔗 **Linking** → 🌍 **GIS Export**

1. **Input:** A scanned or photographed map with red plot boundaries and reference numbers.  
2. **Preprocessing:**  
    - Convert image to HSV color space  
    - Isolate red pixels (lines & text)  
    - Remove noise and skeletonize lines  
3. **Segmentation:** Use Segment Anything Model (SAM) to detect plot polygons.  
4. **OCR Detection:** Use PaddleOCR to read red reference numbers from the image.  
5. **Data Linking:** Match each detected polygon to its corresponding reference number.  
6. **Export:** Save as GeoPackage (`.gpkg`) or shapefile for GIS systems.  

---

## Visual Overview

| Step | Image | Description |
|------|-------|-------------|
| Input Map | <img src="https://github.com/user-attachments/assets/a7abbace-4d08-41d6-8cbe-039e0e8624b0" width="300"/> | Original scanned map with red plot boundaries and reference numbers |
| Skeletonized Image | <img src="https://github.com/user-attachments/assets/336357be-0039-4163-806d-84268278eb28" width="300"/> | Red lines thinned and cleaned for precise segmentation |
| Extracted Red Lines & Numbers | <img src="https://github.com/user-attachments/assets/970ea435-272c-4207-b53c-6e709c8c9f6b" width="300"/> | Boundaries + OCR-detected reference numbers |
| Example OCR Output | <img src="https://github.com/user-attachments/assets/3165a478-2940-4e40-8716-1b341b58e060" width="100"/> | Single extracted reference number for demonstration |
| Final GIS Output | <img src="https://github.com/user-attachments/assets/d50020a3-431d-419f-8ce6-4424f78728b7" width="300"/> | Exported geospatial map with linked boundaries and reference numbers |

---

## 🔧 Requirements

- Python 3.10  
- Anaconda (recommended for environment management)  
- NVIDIA GPU with CUDA 11.8 support (required for GPU acceleration with PyTorch and PaddlePaddle)  

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Consolelives/Map-Seg-Paddle-Linker.git
cd Map-Seg-Paddle-Linker

```
### 2. Create environment
```
conda create -n map_seg python=3.10 -y
conda activate map_seg
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```

🚀 Usage
You can run each step individually via notebooks, or execute the full pipeline at once.

Full pipeline:
```
jupyter notebook Run_all_Process.ipynb
```

### Step-by-step notebooks:

1. IMAGE_PROCESSING.ipynb – Isolate red boundaries & numbers
2. PADDLE_OCR.ipynb – Detect and read reference numbers
3. SAM_MODELS.ipynb – Segment plots into polygons
4. COMBINE_DATA.ipynb – Link polygons with reference numbers
5. GEOPANDAS.ipynb – Export results as GIS-ready data

🌍 Why This Matters
Manual digitization of maps is time-consuming and error-prone. This project automates the process with AI, making it:
Faster (minutes instead of hours)
More accurate (OCR + AI segmentation)
GIS-ready from the start

Keywords: Computer Vision, OCR, PaddleOCR, SAM, GIS Automation, Map Segmentation, Boundary Detection, Land Registry, Geospatial AI

📬 Contact
I can adapt this pipeline to your specific maps, formats, and accuracy requirements.
📧 Email: olalekanoyeleye@yahoo.com
🔗 LinkedIn: https://www.linkedin.com/in/olalekanoyeleye/

🙌 Acknowledgements
Ultralytics
Meta AI – Segment Anything
PaddleOCR
PyTorch
GeoPandas
