# Satellite Image Change Detection

Binary satellite image change detection using bi-temporal imagery and deep learning.

This project investigates how to detect changes between two satellite images of the same area taken at different times. The final system predicts a binary change mask indicating where changes occurred.

The project follows a full Data Science workflow:

* problem definition,
* data exploration,
* feature engineering,
* classical computer vision baselines,
* machine learning baselines,
* deep learning segmentation,
* API deployment,
* Docker-based inference service.

---

## Project Goal

The goal of this project is to answer the following question:

> Given a satellite image before change and a satellite image after change, can we automatically detect where meaningful changes occurred?

The task is formulated as a **binary image segmentation problem**:

* `0` — no change,
* `1` — change.

The main use case is building-level change detection in satellite imagery. Such systems can support:

* urban growth monitoring,
* infrastructure monitoring,
* disaster assessment,
* environmental monitoring,
* automated Earth Observation workflows.

---

## Dataset

This project uses the **LEVIR-CD dataset**, a commonly used benchmark dataset for remote sensing change detection.

The dataset contains pairs of satellite images:

```text
A/      before image
B/      after image
label/  binary change mask
```

Expected local structure:

```text
data/raw/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

The dataset is **not included in this repository** because satellite imagery files are large.
Download the dataset separately and place it in `data/raw/`.

---

## Repository Structure

```text
satellite-image-change-detection/

├── api/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_ssim_baseline.ipynb
│   ├── 04_ml_baseline_random_forest.ipynb
│   ├── 05_patch_based_change_detection.ipynb
│   └── 06_unet_change_detection.ipynb
│
├── outputs/
│   └── best_unet_model.keras
│
├── src/
│
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Methodology

The project was developed incrementally, starting from simple baselines and moving toward more advanced models.

The full modeling pipeline:

```text
Classical CV Difference
        ↓
SSIM Baseline
        ↓
Pixel-Level Random Forest
        ↓
Patch-Based Random Forest
        ↓
U-Net Segmentation Model
        ↓
FastAPI + Docker Deployment
```

This progression makes it possible to understand why each modeling step was necessary.

---

## Notebook Overview

### Notebook 01 — Exploratory Data Analysis

The goal of the first notebook is to understand the dataset structure and inspect the raw data.

Main tasks:

* load before/after image pairs,
* load ground truth masks,
* verify mask values,
* visualize examples,
* understand the binary segmentation target.

Key observation:

```text
0   → unchanged pixel
255 → changed pixel
```

The task is highly imbalanced because most pixels usually belong to the `no change` class.

---

### Notebook 02 — Classical Computer Vision Baseline

The first baseline uses simple image differencing:

```text
difference = |after - before|
```

Pipeline:

1. convert images to grayscale,
2. compute absolute pixel difference,
3. apply thresholding,
4. apply morphology,
5. compare with ground truth.

This method is simple but sensitive to:

* illumination differences,
* shadows,
* vegetation,
* roads,
* small image misalignment.

Result:

```text
IoU ≈ 0.0367
```

Conclusion:

Simple pixel differencing is not robust enough for satellite image change detection.

---

### Notebook 03 — SSIM Baseline

The second baseline uses Structural Similarity Index Measure.

SSIM compares local image structure instead of raw pixel values. The expectation was that it would be more robust than pixel differencing.

However, on this dataset, SSIM still produced many false positives.

Result:

```text
IoU ≈ 0.0420
```

Conclusion:

Classical similarity-based methods are still insufficient for building-level semantic change detection.

---

### Notebook 04 — Pixel-Level Random Forest

The next step was a supervised machine learning baseline.

A Random Forest classifier was trained to classify pixels as:

```text
0 → no change
1 → change
```

Hand-crafted features included:

* RGB absolute difference,
* grayscale difference,
* blurred difference,
* local mean,
* local standard deviation,
* Laplacian edge difference,
* gradient magnitude difference.

Result:

```text
IoU ≈ 0.0946
```

Conclusion:

Pixel-level ML improves over simple CV baselines, but still lacks spatial context.

---

### Notebook 05 — Patch-Based Random Forest

To add local spatial context, the next model used patches instead of individual pixels.

Each feature vector contained:

* a `15×15` patch from the before image,
* a `15×15` patch from the after image,
* a `15×15` patch of absolute difference.

This allowed the model to observe local structures such as roofs, roads, vegetation, and building shapes.

The best threshold-tuned Patch RF model achieved:

```text
Precision ≈ 0.3032
Recall    ≈ 0.4669
F1        ≈ 0.3007
IoU       ≈ 0.1965
```

Conclusion:

Patch-based features significantly improve performance compared to pixel-level features.

---

### Notebook 06 — U-Net Change Detection

The final model is a U-Net segmentation network.

Input:

```text
before RGB image + after RGB image
```

Shape:

```text
256 × 256 × 6
```

Output:

```text
256 × 256 × 1 binary change mask
```

The model was trained using a combined Binary Cross-Entropy + Dice loss.

Average results on 20 positive test crops:

| Method | Threshold | Precision | Recall |     F1 |    IoU |
| ------ | --------: | --------: | -----: | -----: | -----: |
| U-Net  |       0.1 |    0.4151 | 0.6063 | 0.4507 | 0.3314 |

Conclusion:

U-Net achieved the best performance among all tested approaches.

---

## Model Comparison

| Method                  | Precision | Recall |     F1 |    IoU |
| ----------------------- | --------: | -----: | -----: | -----: |
| Classical CV Difference |    0.0398 | 0.4753 | 0.0682 | 0.0367 |
| SSIM Baseline           |    0.0426 | 0.7433 | 0.0806 | 0.0420 |
| Pixel RF                |    0.1478 | 0.2083 | 0.1729 | 0.0946 |
| Patch RF                |    0.3032 | 0.4669 | 0.3007 | 0.1965 |
| U-Net                   |    0.4151 | 0.6063 | 0.4507 | 0.3314 |

The U-Net model clearly achieved the best overall performance.

---

## Key Findings

### 1. Simple image differencing is not enough

Pixel differences are highly sensitive to:

* lighting,
* shadows,
* roads,
* vegetation,
* small image shifts.

### 2. Pixel-level ML lacks context

Random Forest on pixel features improved results, but the model often confused visual differences with actual building changes.

### 3. Spatial context matters

Patch-based Random Forest improved performance because the model could analyze local neighborhoods.

### 4. Deep learning performs best

U-Net achieved the strongest results because it learns hierarchical spatial features directly from data.

---

## Deployment

The best-performing model was deployed using FastAPI.

The API exposes multiple endpoints for inference.

### Decision thresholds

The API uses two separate thresholds:

```text
Model threshold = 0.1
Image-level change threshold = 0.5%
---

## API Endpoints

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "outputs/best_unet_model.keras"
}
```

---

### Basic U-Net Prediction

```text
POST /predict/unet
```

Input:

* `before` image,
* `after` image,
* `threshold`.

This endpoint resizes the full input image to `256×256`.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict/unet" \
  -F "before=@data/raw/test/A/test_112.png" \
  -F "after=@data/raw/test/B/test_112.png" \
  -F "threshold=0.1"
```

Output:

```json
{
  "model": "U-Net",
  "threshold": 0.1,
  "image_size": 256,
  "change_detected": false,
  "changed_area_percent": 0.2762,
  "probability_min": 0.0000008,
  "probability_max": 0.3220,
  "probability_mean": 0.0020,
  "mask_png_base64": "..."
}
```

---

### Basic Mask Endpoint

```text
POST /predict/unet/mask
```

Returns the predicted binary mask directly as a PNG image.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict/unet/mask" \
  -F "before=@data/raw/test/A/test_112.png" \
  -F "after=@data/raw/test/B/test_112.png" \
  -F "threshold=0.1" \
  --output predicted_mask.png
```

---

## Tiled Inference

The model was trained on `256×256` crops.
For full-size `1024×1024` images, resizing the entire image to `256×256` can make buildings too small.

To solve this, the API also supports tiled inference.

The full image is split into `256×256` tiles, each tile is processed by U-Net, and the predicted masks are stitched back into a full-resolution mask.

---

### Tiled JSON Prediction

```text
POST /predict/unet/tiled
```

Example:

```bash
curl -X POST "http://127.0.0.1:8001/predict/unet/tiled" \
  -F "before=@data/raw/test/A/test_112.png" \
  -F "after=@data/raw/test/B/test_112.png" \
  -F "threshold=0.1"
```

Example response:

```json
{
  "model": "U-Net",
  "inference_mode": "tiled",
  "threshold": 0.1,
  "tile_size": 256,
  "original_image_shape": [1024, 1024, 3],
  "change_detected": true,
  "changed_area_percent": 0.7549,
  "probability_min": 0.0000000046,
  "probability_max": 0.9900,
  "probability_mean": 0.0052,
  "mask_png_base64": "..."
}
```

---

### Tiled Mask Endpoint

```text
POST /predict/unet/tiled/mask
```

Returns a full-resolution PNG change mask.

Example:

```bash
curl -X POST "http://127.0.0.1:8001/predict/unet/tiled/mask" \
  -F "before=@data/raw/test/A/test_112.png" \
  -F "after=@data/raw/test/B/test_112.png" \
  -F "threshold=0.1" \
  --output predicted_mask_tiled.png
```

---

## Docker Deployment

The API can be run inside a Docker container.

### Build Docker Image

```bash
docker build -t satellite-change-api .
```

If Git metadata causes build issues, use:

```bash
BUILDX_GIT_INFO=0 docker build -t satellite-change-api .
```

### Run Container

```bash
docker run -p 8000:8000 satellite-change-api
```

If port `8000` is already in use:

```bash
docker run -p 8001:8000 satellite-change-api
```

### Open API Documentation

```text
http://127.0.0.1:8000/docs
```

or, if using port `8001`:

```text
http://127.0.0.1:8001/docs
```

---

## Installation Without Docker

Create virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run API locally:

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Requirements

Main libraries used:

* Python
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* FastAPI
* Uvicorn
* Docker

---

## Limitations

The current project is a strong prototype, but it has several limitations:

1. **Dataset-specific model**

   The U-Net model was trained on LEVIR-CD and may not generalize directly to other satellite datasets.

2. **Binary change detection only**

   The model detects whether change occurred, but does not classify the semantic type of change.

3. **Limited training time**

   The U-Net was trained for a relatively small number of epochs due to local hardware limitations.

4. **False positives remain**

   The model may still confuse real changes with shadows, vegetation, roads, or illumination differences.

5. **No geospatial metadata processing**

   This version works with image files, not georeferenced satellite products such as GeoTIFF or Sentinel `.SAFE` products.

---

## Future Work

Possible improvements:

* train U-Net for more epochs,
* add data augmentation,
* experiment with focal loss or weighted BCE,
* implement full-image tiled evaluation in notebooks,
* compare U-Net with more advanced architectures,
* add semantic change classification,
* support GeoTIFF input,
* use Sentinel-1 SAR or Sentinel-2 data,
* deploy to a cloud service,
* add a frontend visualization dashboard.

---

## Project Summary

This project demonstrates an end-to-end Data Science workflow for satellite image change detection.

The work started with simple classical computer vision baselines and progressively moved toward more advanced models. The experiments showed that spatial context is essential for robust change detection.

The final U-Net model achieved the best performance and was deployed behind a FastAPI endpoint with Docker support.

Overall, the project demonstrates how machine learning and deep learning can be used to automate satellite image monitoring workflows.
