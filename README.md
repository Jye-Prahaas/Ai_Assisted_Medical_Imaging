# AI Medical Imaging

An AI-assisted medical imaging research prototype for chest X-ray
classification, CT-volume segmentation, and visual model explainability.
The application combines a DenseNet121 classifier, a MONAI 3D U-Net,
Grad-CAM visualizations, a FastAPI backend, a React frontend,
Docker-based local deployment, audit logging, and Orthanc integration.

> **Research-use notice:** This project is an educational/research
> prototype. It is not a medical device and must not be used to
> diagnose, treat, or make clinical decisions. Model outputs require
> review by qualified medical professionals.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Key Features](#key-features)
-   [System Architecture](#system-architecture)
-   [Technology Stack](#technology-stack)
-   [AI Models](#ai-models)
-   [Dataset and Preprocessing](#dataset-and-preprocessing)
-   [Model Evaluation](#model-evaluation)
-   [Project Structure](#project-structure)
-   [Prerequisites](#prerequisites)
-   [Run Locally with Docker](#run-locally-with-docker)
-   [Run the Backend Without Docker](#run-the-backend-without-docker)
-   [API Documentation](#api-documentation)
-   [Frontend](#frontend)
-   [Orthanc Integration](#orthanc-integration)
-   [Audit Logging](#audit-logging)
-   [Configuration](#configuration)
-   [Limitations](#limitations)
-   [Future Improvements](#future-improvements)
-   [Responsible Use](#responsible-use)

------------------------------------------------------------------------

## Project Overview

AI Medical Imaging provides a unified interface for two separate
medical-imaging workflows:

1.  **Chest X-ray classification:** DenseNet121 predicts the
    classification result and provides a Grad-CAM visualization to
    highlight image regions that influenced the model.
2.  **3D CT analysis:** A MONAI 3D U-Net processes NIfTI CT volumes or
    DICOM series converted to NIfTI, produces a 3D segmentation mask,
    and provides slice-level CT and Grad-CAM visualizations.

The backend exposes these capabilities through FastAPI. The frontend
communicates with the API to upload scans, browse CT slices, inspect
segmentation and heatmap overlays, and access review/audit
functionality.

## Key Features

-   **Chest X-ray classification** using a trained DenseNet121 model.
-   **Chest X-ray Grad-CAM** visualization returned as a PNG overlay.
-   **3D CT segmentation** using a MONAI 3D U-Net.
-   **NIfTI upload support** for `.nii` and `.nii.gz` volumes.
-   **DICOM series upload** through a ZIP archive containing DICOM
    slices.
-   **CT slice viewer** with slice navigation.
-   **Segmentation overlay** to display the predicted mask on CT slices.
-   **3D CT Grad-CAM** heatmap endpoint for slice-level visualization.
-   **Reviewer decisions** with approved/rejected status and comments.
-   **Audit log** and audit-integrity verification endpoint.
-   **Orthanc integration** with background result monitoring.
-   **Docker Compose deployment** for the backend and frontend.

## System Architecture

``` mermaid
flowchart TD
    U[User] --> F[React Frontend]
    F --> API[FastAPI Backend]

    API --> C[DenseNet121 Classifier]
    API --> S[MONAI 3D U-Net]
    API --> D[DICOM / NIfTI Processing]

    C --> X[Classification Result]
    C --> XG[Chest X-ray Grad-CAM]

    S --> M[3D Segmentation Mask]
    S --> TG[3D CT Grad-CAM]

    API --> V[CT Slice and Overlay Endpoints]
    API --> A[Audit Logging]
    API --> O[Orthanc Integration]

    X --> F
    XG --> F
    M --> F
    TG --> F
    V --> F
    A --> F
    O --> F
```

### CT processing flow

``` mermaid
flowchart LR
    I[Upload NIfTI or DICOM ZIP] --> P[Volume Preprocessing]
    P --> N[3D U-Net Sliding-Window Inference]
    N --> R[Segmentation Probabilities]
    R --> T[Predicted Mask]
    N --> G[3D Grad-CAM]
    T --> V[Slice and Mask Endpoints]
    G --> H[Grad-CAM Slice Endpoint]
    V --> UI[Frontend CT Viewer]
    H --> UI
```

### Request lifecycle

1.  The frontend sends a scan to the FastAPI backend.
2.  The backend validates the upload and prepares a temporary input
    file.
3.  The appropriate model performs inference.
4.  CT image, mask, and Grad-CAM arrays are stored in the backend's
    in-memory scan cache.
5.  The `/predict` response contains JSON-safe metadata and a scan ID;
    image arrays are not serialized into JSON.
6.  The frontend uses the scan ID to request CT slices, mask overlays,
    and Grad-CAM overlays.

The scan cache is in memory and is limited to the latest three scans by
default. Restarting the backend clears this cache.

## Technology Stack

  Area                     Technology
  ------------------------ -----------------------------------
  Backend API              FastAPI
  API server               Uvicorn
  Chest X-ray model        PyTorch, DenseNet121
  3D segmentation          PyTorch, MONAI 3D U-Net
  Image processing         OpenCV, Pillow, NumPy
  Medical image handling   NiBabel, DICOM processing service
  Explainability           Grad-CAM
  Frontend                 React
  Containerization         Docker, Docker Compose
  PACS integration         Orthanc
  Audit records            Project audit service

## AI Models

### 1. Chest X-ray classification --- DenseNet121

The classification workflow uses a DenseNet121 model trained for binary
chest X-ray classification.

-   Input size: 224 × 224.
-   Loss: `BCEWithLogitsLoss` with positive-class weighting.
-   Optimizer: Adam.
-   Learning rate used during training: `0.001`.
-   Training used mixed precision where supported.
-   A patient-level split was used to reduce patient overlap between
    training, validation, and test sets.

The classification endpoint returns a prediction result as JSON. The
separate Grad-CAM endpoint returns a PNG visualization with prediction
and probability in response headers.

### 2. 3D CT segmentation --- MONAI 3D U-Net

The CT segmentation workflow uses a 3D U-Net implemented with MONAI.

Model configuration:

  Parameter              Value
  ---------------------- --------------------------
  Spatial dimensions     3
  Input channels         1
  Output channels        2
  Channels               `(16, 32, 64, 128, 256)`
  Strides                `(2, 2, 2, 2)`
  Residual units         2
  Normalization          Instance
  Dropout                0.1
  Inference patch size   96 × 96 × 96

Preprocessing configuration used during training/inference:

-   Target spacing: `(1.5, 1.5, 2.0)`.
-   CT intensity range: `-1000` to `400` HU.
-   Foreground cropping with margin 8.
-   Spatial padding to `96 × 96 × 96`.
-   Sliding-window inference for volume prediction.

The trained checkpoint used by the backend is
`models/best_3d_unet_v3.pth`.

### 3. Explainability

The project provides two distinct explainability workflows:

-   **Chest X-ray Grad-CAM:** generated by the DenseNet121
    classification workflow.
-   **3D CT Grad-CAM:** generated from the 3D U-Net workflow and served
    per CT slice.

Grad-CAM is a model-interpretability visualization, not proof that a
highlighted region is medically meaningful or that a prediction is
correct.

## Dataset and Preprocessing

### Chest X-ray dataset

The classification model was trained using a preprocessed RSNA Pneumonia
dataset distribution.

Reported dataset summary:

-   26,684 image records.
-   26,684 corresponding mask records.
-   Metadata contained 30,227 rows.
-   Binary target distribution: 20,672 negative and 6,012 positive image
    records.

Patient-level split:

  Split          Records
  ------------ ---------
  Training        18,678
  Validation       4,003
  Test             4,003

The training split contained 4,208 positive records; the validation and
test splits each contained 902 positive records.

### CT dataset

The 3D segmentation model was trained using Medical Segmentation
Decathlon Task06_Lung data.

CT preprocessing includes resampling, intensity scaling, foreground
cropping, and spatial padding. The inference service applies the
corresponding preprocessing pipeline before sliding-window prediction.

> Dataset access and redistribution are governed by the original dataset
> providers' terms and licenses. Do not commit patient data or
> restricted medical images to this repository.

## Model Evaluation

### DenseNet121 classification

Recorded test metrics after the five-epoch training run:

  Metric        Result
  ----------- --------
  Accuracy      0.7217
  Precision     0.4377
  Recall        0.8259
  F1-score      0.5722
  ROC-AUC       0.8345

A threshold analysis identified approximately `0.6123` as a threshold
giving precision around `0.500`, recall around `0.701`, and F1 around
`0.584` on the evaluated data.

These are experiment results from the specified dataset split, not
estimates of clinical performance.

### 3D U-Net segmentation

The best sampled validation Dice recorded during training was `0.3982`
at epoch 250.

A separate full-volume validation experiment across 13 scans reported:

    Probability threshold   Mean Dice
  ----------------------- -----------
                      0.3      0.3891
                      0.4      0.4027
                      0.5      0.4141
                      0.6      0.4247
                      0.7      0.4350

These results are experimental and depend on preprocessing,
thresholding, and validation setup. They should not be interpreted as
clinical validation.

## Project Structure

``` text
Ai_Medical_Imaging/
│
├── app/
│   └──                  # Frontend application
│
├── configs/
│   └──                  # Configuration files
│
├── data/
│   ├── docker_runtime/
│   └──                  # Runtime data; keep datasets out of Git
│
├── models/
│   ├── classifier.py
│   ├── segmentation3d.py
│   ├── densenet121.pth
│   └── best_3d_unet_v3.pth
│
├── notebooks/
│   └──                  # Training and experimentation notebooks
│
├── reports/
│   └──                  # Evaluation and experiment reports
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   └── services/
│       ├── classification_service.py
│       ├── Grad_cam.py
│       ├── segmentation3d_service.py
│       ├── dicom_service.py
│       ├── orthanc_service.py
│       ├── orthanc_worker.py
│       ├── upload_to_orthanc.py
│       └── audit.py
│
├── tests/
│
├── requirements.txt
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
└── README.md
```

The tree is a high-level project map. Exact files may vary as the
project evolves. Do not commit model checkpoints, large datasets,
generated medical images, or runtime databases unless there is a
deliberate, appropriate reason.

## Prerequisites

For local Docker deployment:

-   Windows, macOS, or Linux.
-   Docker Desktop or Docker Engine with Docker Compose.
-   Git (if cloning the repository).
-   Model checkpoint files placed in the expected `models/` directory.

The backend Dockerfile uses Python 3.12 and installs CPU PyTorch wheels.
The currently verified local API health response reported `device: cpu`.

## Run Locally with Docker

Run the commands from the project root directory.

### 1. Build the containers

``` bash
docker compose build
```

To build only the backend:

``` bash
docker compose build backend
```

To build only the frontend:

``` bash
docker compose build frontend
```

### 2. Start the application

``` bash
docker compose up -d
```

### 3. Check running containers

``` bash
docker compose ps
```

### 4. Inspect backend logs

``` bash
docker compose logs --tail=50 backend
```

### 5. Check backend health

Open:

-   `http://localhost:8000/health`
-   `http://localhost:8000/docs`

A successful health response should resemble:

``` json
{
  "status": "healthy",
  "device": "cpu",
  "classifier_loaded": true,
  "segmentation_loaded": true
}
```

### 6. Open the frontend

Open `http://localhost:5173` in a browser.

### 7. Stop the application

``` bash
docker compose down
```

### Rebuild after source-code changes

Backend changes:

``` bash
docker compose build --no-cache backend
docker compose up -d --force-recreate backend
```

Frontend changes:

``` bash
docker compose build frontend
docker compose up -d --force-recreate frontend
```

## Run the Backend Without Docker

The backend is configured for Python 3.12 in the Docker image. For a
local Python environment, use a compatible Python 3.12 installation.

Create and activate a virtual environment in PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API from the project root:

``` powershell
python -m uvicorn src.api.main:app --reload
```

The API is then available at:

-   `http://localhost:8000`
-   `http://localhost:8000/docs`

The model checkpoints must exist at the paths configured in
`src/api/main.py`.

## API Documentation

Interactive OpenAPI documentation is available at:

`http://localhost:8000/docs`

### Main endpoints

  ---------------------------------------------------------------------------
  Method                  Endpoint                    Purpose
  ----------------------- --------------------------- -----------------------
  GET                     `/`                         API information

  GET                     `/health`                   Backend and model
                                                      health

  POST                    `/predict/classification`   Chest X-ray
                                                      classification

  POST                    `/predict/gradcam`          Chest X-ray Grad-CAM
                                                      PNG

  POST                    `/predict`                  NIfTI CT segmentation

  POST                    `/predict/dicom-series`     DICOM ZIP upload and CT
                                                      segmentation

  GET                     `/slice/{scan_id}/{z}`      CT slice PNG

  GET                     `/mask/{scan_id}/{z}`       Segmentation mask
                                                      overlay PNG

  GET                     `/gradcam/{scan_id}/{z}`    3D CT Grad-CAM overlay
                                                      PNG

  POST                    `/decision`                 Save reviewer decision

  GET                     `/audit`                    Retrieve audit records

  GET                     `/audit/verify`             Verify audit integrity

  GET                     `/orthanc/results`          Retrieve Orthanc
                                                      processing results
  ---------------------------------------------------------------------------

### NIfTI segmentation request

`POST /predict`

-   Content type: `multipart/form-data`
-   Form field: `file`
-   Accepted extensions: `.nii`, `.nii.gz`

The response contains metadata such as `scan_id`, volume shape, slice
count, tumour voxel count, estimated volume, confidence, and selected
slice indices.

Example response shape:

``` json
{
  "scan_id": "example123456",
  "filename": "scan.nii.gz",
  "shape": [128, 128, 96],
  "num_slices": 96,
  "tumour_detected": true,
  "tumour_voxels": 1000,
  "tumour_volume_ml": 4.32,
  "confidence": 0.85,
  "first_tumour_slice": 20,
  "last_tumour_slice": 42,
  "best_slice": 31,
  "device": "cpu"
}
```

The values above are illustrative only; actual results depend on the
uploaded scan and model output.

The image, mask, and Grad-CAM arrays are kept in the backend's in-memory
cache. They are retrieved through the slice, mask, and Grad-CAM
endpoints using the returned `scan_id`.

### DICOM series request

`POST /predict/dicom-series`

-   Content type: `multipart/form-data`
-   Form field: `file`
-   Accepted input: ZIP archive containing DICOM slices.
-   Current archive limits: 2,000 files and 500 MB extracted size.

The DICOM service reads the series, converts it to a NIfTI volume for
inference, and returns the same style of scan metadata.

### Slice, mask, and Grad-CAM requests

After receiving a `scan_id`, request:

``` text
GET /slice/{scan_id}/{z}
GET /mask/{scan_id}/{z}
GET /gradcam/{scan_id}/{z}
```

Each endpoint returns a PNG image for the requested slice index `z`.

### Reviewer decision request

`POST /decision`

Example JSON body:

``` json
{
  "scan_id": "example123456",
  "decision": "approved",
  "reviewer": "Reviewer",
  "comment": "Reviewed for demonstration."
}
```

The `decision` value must be `approved` or `rejected`.

### Audit endpoints

``` text
GET /audit
GET /audit/verify
```

The audit list endpoint accepts a `limit` query parameter. The integrity
endpoint reports whether the stored audit chain passes verification.

## Frontend

The React frontend is served by the frontend Docker container on port
`5173`.

The interface supports:

-   NIfTI CT upload.
-   DICOM ZIP upload.
-   CT slice navigation.
-   Segmentation-mask overlay.
-   3D CT Grad-CAM overlay.
-   Reviewer decision workflow.
-   Audit log viewing.
-   Orthanc result viewing.

The chest X-ray classification and Grad-CAM routes remain available in
the backend API. The main dashboard is focused on the CT workflow and
does not need a separate chest X-ray upload section.

## Orthanc Integration

The backend includes an Orthanc service integration and a background
worker that checks for processing results.

The Docker Compose configuration uses:

``` text
ORTHANC_URL=http://host.docker.internal:8042
```

This configuration assumes Orthanc is reachable from the Docker host at
port `8042`. Adjust the environment variable if Orthanc is hosted
elsewhere.

Orthanc result summaries are exposed through:

`GET /orthanc/results`

Orthanc must be separately installed and running; starting this
project's Docker Compose services does not automatically provision an
Orthanc server.

## Audit Logging

Reviewer decisions are recorded through the audit service. The API
supports:

-   Recording an approval or rejection.
-   Associating the decision with a scan ID and filename.
-   Recording reviewer identity and an optional comment.
-   Retrieving audit records.
-   Verifying audit integrity.

The audit system is intended for project traceability and demonstration.
It is not a substitute for a validated clinical audit or hospital
information system.

## Configuration

Important backend settings are currently defined in `src/api/main.py`:

  Setting                            Current value
  ---------------------------------- ----------------------------------
  Classifier checkpoint              `models/densenet121.pth`
  Segmentation checkpoint            `models/best_3d_unet_v3.pth`
  Device selection                   CUDA if available, otherwise CPU
  Target spacing                     `(1.5, 1.5, 2.0)`
  In-memory scan cache               3 scans
  Maximum DICOM files                2,000
  Maximum extracted DICOM ZIP size   500 MB

The backend health endpoint reports the active inference device and
whether each model loaded successfully.

## Limitations

-   This is a research and demonstration prototype, not a clinically
    validated system.
-   Model performance is limited by the training data, labels,
    preprocessing, and evaluation setup.
-   The classification model's reported precision and recall show that
    false positives and false negatives remain possible.
-   3D segmentation results can vary with scan orientation, spacing,
    preprocessing, and threshold selection.
-   Grad-CAM visualizations are approximate explanations and should not
    be interpreted as confirmed disease locations.
-   The current scan cache is in memory, limited to three scans, and
    cleared when the backend restarts.
-   CPU inference may be slow for large CT volumes.
-   Orthanc must be configured and operated separately.
-   The local Docker deployment has been tested; a public cloud
    deployment requires separate configuration, security review, and
    testing.

## Future Improvements

-   Validate models on independent, representative datasets.
-   Improve segmentation performance and investigate preprocessing/label
    alignment.
-   Add persistent, access-controlled scan storage.
-   Add authentication, authorization, and secure upload controls.
-   Add structured monitoring, error tracking, and inference performance
    metrics.
-   Improve GPU inference support and optimize large-volume processing.
-   Expand automated tests for API endpoints and medical-image input
    validation.
-   Complete and document a secure cloud deployment.
-   Conduct clinical and regulatory review before considering any
    real-world medical use.

## Responsible Use

-   Use only datasets for which you have appropriate access and usage
    rights.
-   Do not upload identifiable patient information to public demo
    environments.
-   Keep model checkpoints, patient data, DICOM files, NIfTI scans, and
    runtime databases out of public source control unless explicitly
    permitted and properly protected.
-   Treat all predictions and visualizations as experimental outputs.
-   Do not use this project as the sole basis for medical decisions.
