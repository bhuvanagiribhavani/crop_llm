"""
================================================================================
FASTAPI SERVER - CROP ANALYTICS BACKEND
================================================================================
REST API for connecting React frontend with the U-Net prediction model.

Endpoints:
    POST /predict - Accept image and return segmentation mask + insights
    GET /health - Health check endpoint
    GET /classes - Get class information
    GET /predict/demo - Demo prediction data

Author: Crop Analytics Project
Date: 2026
================================================================================
"""

import os
import sys
import io
import json
import base64
import traceback
import pathlib
import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# PATH SETUP — Ensures imports & relative paths work after folder restructure
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(1, str(PROJECT_ROOT / "database"))
os.chdir(PROJECT_ROOT)          # so 'outputs/', 'frontend/' etc. resolve correctly

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import prediction utilities
from model import get_model

# ============================================================================
# APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title="🌾 Crop Analytics API",
    description="REST API for U-Net crop segmentation and land cover analysis",
    version="2.0.0",
)

# Enable CORS for all origins (same as Flask setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================
UPLOAD_FOLDER = 'outputs/uploads'
PREDICTION_FOLDER = 'outputs/predictions'
MODEL_PATH = 'outputs/best_model.pth'
IMAGE_SIZE = 256
NUM_CLASSES = 8

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Class names and colors for visualization
CLASS_NAMES = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded Vegetation",
    4: "Crops",
    5: "Scrub/Shrub",
    6: "Built Area",
    7: "Bare Ground"
}

CLASS_COLORS = {
    0: [65, 155, 223],    # Water - Blue
    1: [57, 125, 73],     # Trees - Dark Green
    2: [136, 176, 83],    # Grass - Light Green
    3: [122, 135, 198],   # Flooded Vegetation - Purple-Blue
    4: [228, 150, 53],    # Crops - Orange
    5: [223, 193, 125],   # Scrub/Shrub - Tan
    6: [196, 40, 27],     # Built Area - Red
    7: [165, 155, 143]    # Bare Ground - Gray
}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Global model variable
model = None
device = None


# ============================================================================
# MODEL LOADING & UTILITIES
# ============================================================================

def load_model():
    """Load the trained U-Net model."""
    global model, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=NUM_CLASSES)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠ Warning: Model not found at {MODEL_PATH}, using random weights")

    model.to(device)
    model.eval()
    return model


def preprocess_image(image):
    """Preprocess image for model inference."""
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    img_array = np.array(image).astype(np.float32) / 255.0

    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor


def colorize_mask(mask):
    """Convert class mask to RGB colored image."""
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        colored_mask[mask == class_id] = color

    return colored_mask


def calculate_statistics(mask):
    """Calculate class distribution statistics from mask."""
    total_pixels = mask.size
    stats = {}

    for class_id, class_name in CLASS_NAMES.items():
        pixel_count = int(np.sum(mask == class_id))
        percentage = (pixel_count / total_pixels) * 100
        stats[class_name] = {
            "pixel_count": pixel_count,
            "percentage": round(percentage, 2),
            "area_hectares": round(pixel_count * 0.01, 2)
        }

    return stats


def generate_insight(stats):
    """Generate AI insight based on statistics."""
    sorted_classes = sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
    dominant = sorted_classes[0]

    insights = []

    crops_pct = stats.get('Crops', {}).get('percentage', 0)
    veg_pct = stats.get('Trees', {}).get('percentage', 0) + \
              stats.get('Grass', {}).get('percentage', 0) + \
              stats.get('Flooded Vegetation', {}).get('percentage', 0)

    if crops_pct > 30:
        insights.append(f"🌾 High crop density detected ({crops_pct:.1f}%). Agricultural activity is prominent in this region.")
    elif crops_pct > 10:
        insights.append(f"🌾 Moderate crop coverage ({crops_pct:.1f}%) indicating active farming zones.")
    elif crops_pct > 0:
        insights.append(f"🌱 Low crop density ({crops_pct:.1f}%). Consider expanding agricultural coverage.")

    total_veg = crops_pct + veg_pct + stats.get('Scrub/Shrub', {}).get('percentage', 0)
    if total_veg > 70:
        insights.append(f"🌿 Excellent vegetation coverage ({total_veg:.1f}%). Ecosystem health appears good.")
    elif total_veg > 40:
        insights.append(f"🌿 Moderate vegetation ({total_veg:.1f}%). Consider conservation measures.")

    water_pct = stats.get('Water', {}).get('percentage', 0)
    if water_pct > 5:
        insights.append(f"💧 Water bodies detected ({water_pct:.1f}%). Good irrigation potential.")

    built_pct = stats.get('Built Area', {}).get('percentage', 0)
    if built_pct > 20:
        insights.append(f"🏗️ Significant urbanization ({built_pct:.1f}%). Monitor land use changes.")

    if not insights:
        insights.append(f"📊 Analysis complete. Dominant land cover: {dominant[0]} ({dominant[1]['percentage']:.1f}%)")

    return " ".join(insights)


# ============================================================================
# STARTUP EVENT - Load model when server starts
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load the model when the FastAPI server starts."""
    print("=" * 60)
    print("CROP ANALYTICS - FASTAPI SERVER")
    print("=" * 60)
    load_model()
    print("\n✓ FastAPI Server starting...")
    print("  Endpoints:")
    print("    - GET  /health       : Health check")
    print("    - GET  /classes      : Get class info")
    print("    - POST /predict      : Run prediction")
    print("    - GET  /predict/demo : Demo data")
    print("    - GET  /docs         : Swagger UI (auto-generated)")
    print("    - GET  /redoc        : ReDoc API docs")
    print("\n  Frontend URL: http://localhost:3000")
    print("  API URL: http://localhost:5000")
    print("=" * 60)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def index():
    """Root endpoint with API information."""
    return {
        "name": "🌾 Crop Analytics API",
        "version": "2.0.0 (FastAPI)",
        "status": "running",
        "message": "Welcome to Crop Analytics! Access the dashboard at http://localhost:3000",
        "endpoints": {
            "POST /predict": "Upload image for segmentation",
            "GET /health": "API health check",
            "GET /classes": "Get class definitions",
            "GET /predict/demo": "Get demo prediction",
            "GET /docs": "Swagger UI (interactive API docs)",
            "GET /redoc": "ReDoc API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }


@app.get("/classes")
async def get_classes():
    """Get class information."""
    return {
        "classes": CLASS_NAMES,
        "colors": CLASS_COLORS
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Main prediction endpoint.
    Accepts an image file and returns segmentation mask + insights.
    """
    try:
        # Validate file was uploaded
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image selected")

        # Read and process image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Save original image
        original_filename = f"uploaded_{image.filename}"
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        pil_image.save(original_path)

        # Preprocess for model
        img_tensor = preprocess_image(pil_image)
        img_tensor = img_tensor.to(device)

        # ================================================================
        # IMAGE VALIDATION - Detect non-satellite images
        # ================================================================
        img_array = np.array(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE)))

        # 1. Check color distribution
        r_mean = img_array[:, :, 0].mean()
        g_mean = img_array[:, :, 1].mean()
        b_mean = img_array[:, :, 2].mean()
        r_std = img_array[:, :, 0].std()
        g_std = img_array[:, :, 1].std()
        b_std = img_array[:, :, 2].std()

        # 2. Overall std
        overall_std = img_array.std()

        # 3. Channel correlation
        r_flat = img_array[:, :, 0].flatten().astype(float)
        g_flat = img_array[:, :, 1].flatten().astype(float)
        b_flat = img_array[:, :, 2].flatten().astype(float)

        rg_corr = np.corrcoef(r_flat, g_flat)[0, 1] if r_std > 0 and g_std > 0 else 0
        rb_corr = np.corrcoef(r_flat, b_flat)[0, 1] if r_std > 0 and b_std > 0 else 0
        avg_corr = (abs(rg_corr) + abs(rb_corr)) / 2

        # 4. Unique colors
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))

        # 5. Skin tone indicator
        is_skin_tone_dominant = (r_mean > g_mean > b_mean) and (r_mean - b_mean > 20)

        # 6. Red dominance
        red_dominance = r_mean / (g_mean + 0.01)

        # 7. Texture complexity
        gray = np.mean(img_array, axis=2)
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        texture_complexity = gx + gy

        # Validation
        is_valid_satellite = True
        rejection_reasons = []

        if overall_std < 15:
            is_valid_satellite = False
            rejection_reasons.append("Image appears to be blank or solid color")

        if overall_std > 120:
            is_valid_satellite = False
            rejection_reasons.append("Image has unnatural contrast levels")

        if avg_corr < 0.3 and overall_std > 30:
            is_valid_satellite = False
            rejection_reasons.append("Color patterns don't match satellite imagery")

        color_imbalance = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
        if color_imbalance > 60:
            is_valid_satellite = False
            rejection_reasons.append("Color balance doesn't match natural satellite imagery")

        if is_skin_tone_dominant and r_mean > 120 and red_dominance > 1.15:
            is_valid_satellite = False
            rejection_reasons.append("Image appears to contain human skin tones (not satellite imagery)")

        if unique_colors < 500 and overall_std > 20:
            is_valid_satellite = False
            rejection_reasons.append("Image lacks color diversity typical of satellite imagery")

        avg_brightness = (r_mean + g_mean + b_mean) / 3
        if avg_brightness > 150 and texture_complexity < 15:
            is_valid_satellite = False
            rejection_reasons.append("Image texture doesn't match satellite imagery patterns")

        if red_dominance > 1.3 and r_mean > 100:
            is_valid_satellite = False
            rejection_reasons.append("Red channel dominance suggests non-satellite image")

        if not is_valid_satellite:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"⚠️ Invalid input image detected. {'; '.join(rejection_reasons)}. Please upload Sentinel-2 satellite imagery only (RGB land cover images).",
                    "validation_details": {
                        "overall_std": float(overall_std),
                        "channel_correlation": float(avg_corr),
                        "color_imbalance": float(color_imbalance),
                        "unique_colors": int(unique_colors),
                        "red_dominance": float(red_dominance),
                        "skin_tone_detected": bool(is_skin_tone_dominant),
                        "reasons": rejection_reasons
                    }
                }
            )

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)

            # Compute confidence
            probabilities = torch.softmax(output, dim=1)
            max_probs = probabilities.max(dim=1)[0]
            confidence = max_probs.mean().item()

            # Entropy-based validation
            entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=1).mean().item()

            CONFIDENCE_THRESHOLD = 0.45
            if confidence < CONFIDENCE_THRESHOLD:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": f"⚠️ Low prediction confidence ({confidence:.1%}). This image doesn't appear to be valid Sentinel-2 satellite imagery. Please upload RGB crop/land cover images.",
                        "confidence": confidence,
                        "threshold": CONFIDENCE_THRESHOLD
                    }
                )

            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Generate colored mask
        colored_mask = colorize_mask(prediction)
        mask_image = Image.fromarray(colored_mask)

        # Save prediction mask
        mask_filename = f"mask_{image.filename}"
        mask_path = os.path.join(PREDICTION_FOLDER, mask_filename)
        mask_image.save(mask_path)

        mask_png_path = os.path.join(PREDICTION_FOLDER, 'predicted_mask_colored.png')
        mask_image.save(mask_png_path)

        # Calculate statistics
        stats = calculate_statistics(prediction)

        # Generate insight
        insight = generate_insight(stats)

        # Convert mask to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Convert original image to base64
        buffer_orig = io.BytesIO()
        pil_image.resize((IMAGE_SIZE, IMAGE_SIZE)).save(buffer_orig, format='PNG')
        original_base64 = base64.b64encode(buffer_orig.getvalue()).decode('utf-8')

        # Store prediction for LLM Insights
        pred_id = _save_prediction(stats, insight, [CLASS_NAMES[i] for i in np.unique(prediction)])

        return {
            "success": True,
            "mask_path": mask_path,
            "mask_base64": f"data:image/png;base64,{mask_base64}",
            "original_base64": f"data:image/png;base64,{original_base64}",
            "statistics": stats,
            "insight": insight,
            "confidence": confidence,
            "classes_detected": [CLASS_NAMES[i] for i in np.unique(prediction)]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/demo")
async def predict_demo():
    """Demo endpoint returning sample data for testing."""
    demo_stats = {
        "Water": {"pixel_count": 100, "percentage": 1.5, "area_hectares": 1.0},
        "Trees": {"pixel_count": 500, "percentage": 7.6, "area_hectares": 5.0},
        "Grass": {"pixel_count": 800, "percentage": 12.2, "area_hectares": 8.0},
        "Flooded Vegetation": {"pixel_count": 300, "percentage": 4.6, "area_hectares": 3.0},
        "Crops": {"pixel_count": 2500, "percentage": 38.1, "area_hectares": 25.0},
        "Scrub/Shrub": {"pixel_count": 1500, "percentage": 22.9, "area_hectares": 15.0},
        "Built Area": {"pixel_count": 600, "percentage": 9.2, "area_hectares": 6.0},
        "Bare Ground": {"pixel_count": 256, "percentage": 3.9, "area_hectares": 2.56}
    }
    demo_insight = "🌾 High crop density detected (38.1%). Agricultural activity is prominent in this region. 🌿 Excellent vegetation coverage (85.4%). Ecosystem health appears good."
    demo_classes = ["Water", "Trees", "Grass", "Flooded Vegetation", "Crops", "Scrub/Shrub", "Built Area", "Bare Ground"]

    # Store demo prediction for LLM Insights
    _save_prediction(demo_stats, demo_insight, demo_classes)

    return {
        "success": True,
        "mask_path": "outputs/predictions/sample_predicted_mask.png",
        "statistics": demo_stats,
        "insight": demo_insight,
        "classes_detected": demo_classes
    }


# ============================================================================
# IN-MEMORY PREDICTION STORE  (used by LLM Insights until DB is wired)
# ============================================================================
_prediction_store: dict = {}   # id -> prediction dict
_prediction_counter: int = 0


def _save_prediction(stats: dict, insight: str, classes_detected: list):
    """Persist prediction data in memory and return its ID."""
    global _prediction_counter
    _prediction_counter += 1
    pid = _prediction_counter

    # Derive dominant crop type
    sorted_classes = sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
    crop_type = sorted_classes[0][0] if sorted_classes else "Unknown"

    # Simulated NDVI from vegetation percentages
    trees   = stats.get('Trees', {}).get('percentage', 0)
    crops   = stats.get('Crops', {}).get('percentage', 0)
    grass   = stats.get('Grass', {}).get('percentage', 0)
    flooded = stats.get('Flooded Vegetation', {}).get('percentage', 0)
    ndvi    = round(min((trees * 0.8 + crops * 0.65 + grass * 0.5 + flooded * 0.4) / 100, 0.85), 2)

    # Yield estimation (tons) — same logic as frontend
    crop_pct   = crops
    water_pct  = stats.get('Water', {}).get('percentage', 0)
    grass_pct  = grass
    total_area = 100
    crop_area  = (crop_pct / 100) * total_area
    base_yield = 4.5
    mult = 1.0
    if 5 < water_pct < 20:
        mult += 0.15
    if grass_pct > 10:
        mult -= 0.1
    estimated_yield = round(crop_area * base_yield * mult, 1)

    _prediction_store[pid] = {
        "id": pid,
        "crop_type": crop_type,
        "ndvi": ndvi,
        "estimated_yield_tons": estimated_yield,
        "statistics": stats,
        "insight": insight,
        "classes_detected": classes_detected,
    }
    return pid


# ============================================================================
# LLM INSIGHTS ENDPOINTS
# ============================================================================

@app.get("/predictions")
async def list_predictions():
    """Return all stored predictions (most recent first)."""
    preds = sorted(_prediction_store.values(), key=lambda p: p["id"], reverse=True)
    return {"predictions": preds}


@app.post("/llm-insight/{prediction_id}")
async def llm_insight(prediction_id: int):
    """
    Generate an LLM-style AI insight for a given prediction.

    Currently uses structured placeholder logic.
    In production, replace the body with a call to Mistral-7B via Ollama.
    """
    pred = _prediction_store.get(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    crop_type = pred["crop_type"]
    ndvi      = pred["ndvi"]
    yield_t   = pred["estimated_yield_tons"]

    # ── Structured prompt (reserved for real LLM call) ──────────────
    # prompt = (
    #     f"You are an agricultural AI assistant.\n"
    #     f"Crop type: {crop_type}\n"
    #     f"NDVI: {ndvi}\n"
    #     f"Estimated yield: {yield_t} tons\n"
    #     f"Provide a detailed insight and actionable recommendations."
    # )
    # response = ollama.chat(model="mistral", messages=[{"role":"user","content": prompt}])

    # ── Placeholder deterministic insight ───────────────────────────
    if ndvi >= 0.6:
        health = "excellent"
        health_detail = "The vegetation is dense and thriving, indicating optimal growing conditions and adequate water supply."
    elif ndvi >= 0.4:
        health = "good"
        health_detail = "Vegetation health is satisfactory. Minor improvements in irrigation or soil nutrients could enhance growth."
    elif ndvi >= 0.2:
        health = "moderate"
        health_detail = "Vegetation shows signs of stress. Soil moisture levels and nutrient availability should be assessed."
    else:
        health = "poor"
        health_detail = "Vegetation cover is critically low. Immediate intervention is recommended to prevent crop failure."

    insight_text = (
        f"The dominant land cover in this region is {crop_type} with an NDVI of {ndvi:.2f}, "
        f"indicating {health} vegetation health. {health_detail} "
        f"The estimated yield for the analyzed area is {yield_t} tons across the cropped zone."
    )

    recommendations = []
    if ndvi < 0.4:
        recommendations.append("Increase irrigation frequency to improve soil moisture levels.")
        recommendations.append("Apply nitrogen-based fertilizers to boost vegetative growth.")
    if ndvi < 0.6:
        recommendations.append("Monitor for early signs of pest or disease infestation.")
        recommendations.append("Consider mulching to retain soil moisture during dry spells.")
    if yield_t < 50:
        recommendations.append("Analyze soil composition and amend with organic matter to improve yield potential.")
    recommendations.append("Schedule periodic satellite scans to track vegetation health over time.")
    recommendations.append("Cross-reference NDVI trends with local weather data for precision agriculture decisions.")

    return {
        "prediction_id": prediction_id,
        "crop_type": crop_type,
        "ndvi": ndvi,
        "estimated_yield_tons": yield_t,
        "insight": insight_text,
        "recommendations": recommendations,
    }


# ============================================================================
# SERVE FRONTEND (Production Build)
# ============================================================================
from fastapi.responses import FileResponse

FRONTEND_BUILD = PROJECT_ROOT / "frontend" / "build"

if FRONTEND_BUILD.exists():
    # Serve static files (JS, CSS, images)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_BUILD / "static")), name="frontend-static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve React frontend for any non-API route."""
        file_path = FRONTEND_BUILD / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_BUILD / "index.html"))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "backend"), str(PROJECT_ROOT / "database")],
        reload_excludes=[
            "frontend/*",
            "frontend_backup/*",
            "outputs/*",
            "SEN-2 LULC/*",
            "miniDataSet/*",
            "*.log",
        ],
    )
