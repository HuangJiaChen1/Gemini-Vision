from typing import List, Optional
from pydantic import BaseModel, Field


class RecognitionResult(BaseModel):
    """Result from successful object recognition"""
    object_name: str = Field(description="The friendly name of the recognized object")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)
    description: str = Field(description="A simple, child-friendly description of the object")


class ImageAnalysis(BaseModel):
    """Stage 1: Pure observation - what the AI sees in the image"""
    comprehensive_explanation: str = Field(description="Stream-of-thought analysis describing what the AI sees")
    image_quality: str = Field(description="GOOD, MODERATE, or POOR")
    quality_issues: List[str] = Field(description="List of issues: blur, dark, cropped, obstruction, etc.", default_factory=list)
    detected_objects: List[str] = Field(description="Actual names of objects seen (e.g., Building, Car, Hand)", default_factory=list)
    guidance: str = Field(description="1-2 sentence child-friendly advice to improve the photo", default="")
    # Set by post-processing logic, not by LLM
    confidence_level: str = Field(description="HIGH, MEDIUM, or LOW", default="LOW")
    recommendation: str = Field(description="CLASSIFY, MULTI_SELECT, or GUIDE", default="GUIDE")


class DiagnosticResult(BaseModel):
    """Result when recognition confidence is low"""
    comprehensive_explanation: str = Field(description="A comprehensive summary of the issues and tips")
    issue: str = Field(description="The primary issue: BLUR, TOO_CLOSE, TOO_FAR, LIGHTING, OBSTRUCTION, or UNCLEAR")
    friendly_message: str = Field(description="Child-friendly explanation of the problem")
    guesses: List[str] = Field(description="2-3 guesses about what the object might be")


class DetectedObject(BaseModel):
    """A single detected object in a multi-object image"""
    object_name: str = Field(description="The friendly name of the object")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)
    description: str = Field(description="A simple, child-friendly description of the object")
    box_2d: List[int] = Field(description="Bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000 scale")


class MultiObjectResult(BaseModel):
    """Result when multiple objects are detected in the image"""
    objects: List[DetectedObject] = Field(description="List of detected objects (2-4 objects)")
    message: str = Field(description="A friendly message asking the user to pick one")


class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = Field(description="Whether the request was successful")
    result: Optional[RecognitionResult] = None
    diagnostic: Optional[DiagnosticResult] = None
    multi_object: Optional[MultiObjectResult] = None
    error: Optional[str] = None
