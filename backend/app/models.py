from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import DateTime
from sqlalchemy import Text

from app.database import Base


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    report_name = Column(
        String(100),
        nullable=False
    )

    image_name = Column(
        String(100),
        nullable=False,
        unique=True
    )

    source = Column(
        String(50),
        default="upload"
    )

    status = Column(
        String(50),
        default="Completed"
    )

    changed_pixels = Column(
        Integer,
        default=0
    )

    change_percentage = Column(
        Float,
        default=0.0
    )

    processing_time = Column(
        Float,
        default=0.0
    )

    object_count = Column(
        Integer,
        default=0
    )

    detected_classes = Column(
        Text,
        default="[]"
    )

    before_image = Column(Text)

    after_image = Column(Text)

    overlay_path = Column(Text)

    binary_mask_path = Column(Text)

    before_semantic_path = Column(Text)

    after_semantic_path = Column(Text)

    chart_path = Column(Text)

    json_path = Column(Text)

    pdf_path = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )