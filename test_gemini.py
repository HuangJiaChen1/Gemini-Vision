import json
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# --- Data Models ---
class DetectedObject(BaseModel):
    label: str = Field(description="The name of the detected object")
    description: str = Field(description="A brief description of the object")


class DetectionResult(BaseModel):
    objects: List[DetectedObject]


# --- Main Logic ---
with open("config.json", "r") as f:
    config = json.load(f)

client = genai.Client(
    vertexai=True,
    project=config["project"],
    location=config["location"]
)

image_path = "1.jpg"  # Replace with your image

# Read bytes for Gemini
with open(image_path, "rb") as img_file:
    image_bytes = img_file.read()

# --- Generate Content ---
print("Analyzing image...")
response = client.models.generate_content(
    model=config["model_name"],
    contents=[
        types.Part.from_text(text="Identify and describe the objects in this image. Output a JSON list where each entry contains the 'label' (name of the object) and 'description' (brief description). Use descriptive labels."),
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
    ],
    config=types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=DetectionResult
    )
)

# --- Output Results ---
if response.parsed:
    print(f"\nFound {len(response.parsed.objects)} objects:\n")
    for i, obj in enumerate(response.parsed.objects, 1):
        print(f"{i}. {obj.label}")
        print(f"   Description: {obj.description}\n")
else:
    print("No objects found.")