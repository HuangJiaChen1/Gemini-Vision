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


class ImageDecision(BaseModel):
    """Stage 2: Decision based on the analysis"""
    object_count: int = Field(description="Number of distinct objects detected", ge=0)
    detected_objects: List[str] = Field(description="Names of objects seen in the image")
    confidence_level: str = Field(description="HIGH, MEDIUM, or LOW")
    recommendation: str = Field(description="What to do next: CLASSIFY, MULTI_SELECT, or GUIDE")


class DiagnosticResult(BaseModel):
    """Result when recognition confidence is low"""
    comprehensive_explanation: str = Field(description="A comprehensive explanation of the image quality")
    issue: str = Field(description="The primary issue: BLUR, TOO_CLOSE, TOO_FAR, LIGHTING, OBSTRUCTION, or UNCLEAR")
    friendly_message: str = Field(description="Child-friendly explanation of the problem")
    guesses: List[str] = Field(description="2-3 guesses about what the object might be")
    confidence_of_guesses: List[float] = Field(description="Confidence scores for each guess")


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
