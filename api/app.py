import base64
import io
import os

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response


# =========================
# CONFIG
# =========================

MODEL_PATH = "outputs/best_unet_model.keras"

IMG_SIZE = 256
DEFAULT_THRESHOLD = 0.1
MIN_CHANGED_AREA_PERCENT = 0.5


# =========================
# LOAD MODEL
# =========================

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Satellite Image Change Detection API",
    description="U-Net based binary change detection for before/after satellite images.",
    version="1.0.0"
)


# =========================
# HELPERS
# =========================

def read_image(upload_file: UploadFile) -> np.ndarray:
    """
    Reads uploaded image and returns RGB numpy array.
    """

    image_bytes = upload_file.file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image = np.array(image)

    return image

def read_image_original(upload_file: UploadFile) -> np.ndarray:
    """
    Reads uploaded image without resizing and returns RGB numpy array.
    """

    image_bytes = upload_file.file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = np.array(image)

    return image

def preprocess_images(
    before: np.ndarray,
    after: np.ndarray
) -> np.ndarray:
    """
    Converts before/after RGB images into 6-channel model input.
    """

    before = before.astype(np.float32) / 255.0
    after = after.astype(np.float32) / 255.0

    x = np.concatenate(
        [before, after],
        axis=-1
    )

    x = np.expand_dims(
        x,
        axis=0
    )

    return x


def mask_to_base64(mask: np.ndarray) -> str:
    """
    Converts binary mask to base64 PNG.
    """

    mask_img = (mask * 255).astype(np.uint8)

    pil_img = Image.fromarray(
        mask_img
    )

    buffer = io.BytesIO()

    pil_img.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded

def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """
    Converts binary mask to PNG bytes.
    """

    mask_img = (mask * 255).astype(np.uint8)

    pil_img = Image.fromarray(mask_img)

    buffer = io.BytesIO()

    pil_img.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()

def predict_tiled(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    tile_size: int = IMG_SIZE
):
    """
    Runs U-Net prediction tile-by-tile and stitches the result back.
    Assumes before and after have the same shape.
    """

    h, w, _ = before.shape

    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    before_padded = np.pad(
        before,
        (
            (0, pad_h),
            (0, pad_w),
            (0, 0)
        ),
        mode="reflect"
    )

    after_padded = np.pad(
        after,
        (
            (0, pad_h),
            (0, pad_w),
            (0, 0)
        ),
        mode="reflect"
    )

    hp, wp, _ = before_padded.shape

    proba_full = np.zeros(
        (hp, wp),
        dtype=np.float32
    )

    for y in range(0, hp, tile_size):

        for x in range(0, wp, tile_size):

            before_tile = before_padded[
                y:y+tile_size,
                x:x+tile_size
            ]

            after_tile = after_padded[
                y:y+tile_size,
                x:x+tile_size
            ]

            before_tile = before_tile.astype(np.float32) / 255.0
            after_tile = after_tile.astype(np.float32) / 255.0

            model_input = np.concatenate(
                [before_tile, after_tile],
                axis=-1
            )

            model_input = np.expand_dims(
                model_input,
                axis=0
            )

            pred_tile = model(
                model_input,
                training=False
            ).numpy()[0, :, :, 0]

            proba_full[
                y:y+tile_size,
                x:x+tile_size
            ] = pred_tile

    proba_full = proba_full[:h, :w]

    pred_mask = (
        proba_full > threshold
    ).astype(np.uint8)

    return pred_mask, proba_full

# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {
        "message": "Satellite Image Change Detection API is running.",
        "available_endpoints": [
            "/health",
            "/predict/unet"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict/unet")
def predict_unet(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD)
):
    """
    Predicts binary change mask from before/after satellite images.
    """

    try:
        before_img = read_image(
            before
        )

        after_img = read_image(
            after
        )

        x = preprocess_images(
            before_img,
            after_img
        )

        pred_proba = model(
            x,
            training=False
        ).numpy()[0, :, :, 0]

        pred_mask = (
            pred_proba > threshold
        ).astype(np.uint8)

        changed_area_percent = float(
            pred_mask.sum()
            / pred_mask.size
            * 100
        )

        change_detected = (
            changed_area_percent
            >= MIN_CHANGED_AREA_PERCENT
        )

        mask_base64 = mask_to_base64(
            pred_mask
        )

        return {
            "model": "U-Net",
            "threshold": threshold,
            "image_size": IMG_SIZE,
            "change_detected": change_detected,
            "changed_area_percent": round(
                changed_area_percent,
                4
            ),
            "probability_min": float(
                pred_proba.min()
            ),
            "probability_max": float(
                pred_proba.max()
            ),
            "probability_mean": float(
                pred_proba.mean()
            ),
            "mask_png_base64": mask_base64
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

@app.post("/predict/unet/mask")
def predict_unet_mask(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD)
):
    """
    Returns predicted binary change mask directly as PNG image.
    """

    try:
        before_img = read_image(
            before
        )

        after_img = read_image(
            after
        )

        x = preprocess_images(
            before_img,
            after_img
        )

        pred_proba = model(
            x,
            training=False
        ).numpy()[0, :, :, 0]

        pred_mask = (
            pred_proba > threshold
        ).astype(np.uint8)

        png_bytes = mask_to_png_bytes(
            pred_mask
        )

        return Response(
            content=png_bytes,
            media_type="image/png"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

@app.post("/predict/unet/tiled")
def predict_unet_tiled(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD)
):
    """
    Predicts binary change mask using tiled inference.
    Recommended for full-size images.
    """

    try:
        before_img = read_image_original(before)
        after_img = read_image_original(after)

        if before_img.shape != after_img.shape:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Before and after images must have the same shape."
                }
            )

        pred_mask, pred_proba = predict_tiled(
            before_img,
            after_img,
            threshold=threshold
        )

        changed_area_percent = float(
            pred_mask.sum()
            / pred_mask.size
            * 100
        )

        change_detected = (
            changed_area_percent
            >= MIN_CHANGED_AREA_PERCENT
        )

        mask_base64 = mask_to_base64(
            pred_mask
        )

        return {
            "model": "U-Net",
            "inference_mode": "tiled",
            "threshold": threshold,
            "tile_size": IMG_SIZE,
            "original_image_shape": before_img.shape,
            "change_detected": change_detected,
            "changed_area_percent": round(
                changed_area_percent,
                4
            ),
            "probability_min": float(
                pred_proba.min()
            ),
            "probability_max": float(
                pred_proba.max()
            ),
            "probability_mean": float(
                pred_proba.mean()
            ),
            "mask_png_base64": mask_base64
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

@app.post("/predict/unet/tiled/mask")
def predict_unet_tiled_mask(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD)
):
    """
    Returns tiled U-Net prediction mask directly as PNG.
    """

    try:
        before_img = read_image_original(before)
        after_img = read_image_original(after)

        if before_img.shape != after_img.shape:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Before and after images must have the same shape."
                }
            )

        pred_mask, _ = predict_tiled(
            before_img,
            after_img,
            threshold=threshold
        )

        png_bytes = mask_to_png_bytes(
            pred_mask
        )

        return Response(
            content=png_bytes,
            media_type="image/png"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )