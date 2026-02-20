"""
================================================================================
DATABASE MODELS (ORM)
================================================================================
SQLAlchemy + GeoAlchemy2 models for storing crop prediction results.
================================================================================
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from geoalchemy2 import Geometry

from database import Base


class CropPrediction(Base):
    """
    Stores each prediction made by the U-Net model.

    Columns:
        id               - Auto-increment primary key
        image_name       - Original uploaded filename
        predicted_crop_type - Dominant land cover class
        segmented_area   - Total segmented area in hectares
        ndvi_mean        - Mean NDVI value (if available)
        confidence       - Model prediction confidence (0-1)
        class_distribution - JSON string of class percentages
        insight          - AI-generated insight text
        prediction_date  - Timestamp of prediction
        geometry         - Polygon geometry (SRID 4326) for spatial queries
    """
    __tablename__ = "crop_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_name = Column(String(255), nullable=False)
    predicted_crop_type = Column(String(100), nullable=False)
    segmented_area = Column(Float, nullable=True)
    ndvi_mean = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    class_distribution = Column(Text, nullable=True)       # JSON string
    insight = Column(Text, nullable=True)
    prediction_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    geometry = Column(Geometry("POLYGON", srid=4326), nullable=True)

    def to_dict(self):
        """Convert row to JSON-serializable dict."""
        return {
            "id": self.id,
            "image_name": self.image_name,
            "predicted_crop_type": self.predicted_crop_type,
            "segmented_area": self.segmented_area,
            "ndvi_mean": self.ndvi_mean,
            "confidence": self.confidence,
            "class_distribution": self.class_distribution,
            "insight": self.insight,
            "prediction_date": self.prediction_date.isoformat() if self.prediction_date else None,
            "geometry": None  # Geometry is not directly JSON serializable; use WKT if needed
        }
