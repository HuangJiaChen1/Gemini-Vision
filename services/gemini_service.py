import json
from typing import Union
from google import genai
from google.genai import types
from models.response_models import RecognitionResult, DiagnosticResult, MultiObjectResult, ImageAnalysis


class GeminiService:
    """
    Singleton service for interacting with Google Gemini API
    Provides two-stage prompting: recognition + diagnostics
    """

    _instance = None
    _client = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Gemini client with config from config.json"""
        if self._client is None:
            with open("config.json", "r") as f:
                self._config = json.load(f)

            self._client = genai.Client(
                vertexai=True,
                project=self._config["project"],
                location=self._config["location"]
            )
            print(f"Gemini client initialized with model: {self._config['model_name']}")

    def analyze_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> ImageAnalysis:
        """
        Stage 1: Pure observation - analyze what's in the image.
        Focuses ONLY on describing what it sees, not making decisions.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image

        Returns:
            ImageAnalysis with comprehensive explanation, quality assessment, and issues
        """
        prompt = """You are an intelligent Image Diagnostic Assistant. Your goal is to analyze the *quality*, *composition*, and *clarity* of the photo and describe exactly what you see.

When analyzing an image, follow this "Stream of Thought" format:

1. **Visual Audit (Thinking Aloud):** Start with phrases like "Hmm, I see...", "Looking closely...", or "I notice...". Describe the raw visual data. Is it dark? Blurry? Are there obstructions? What objects are visible? How many distinct objects do you see?
2. **Hypothesis Generation:** Try to guess what the subject(s) are. Use phrases like "I am suggesting maybe...", "This looks like it could be...", or "It seems the user is trying to capture..."
3. **Identify Distortions/Issues:** Explicitly name any problems (e.g., macro blur, low light noise, glass reflection, motion blur, extreme cropping, multiple competing subjects).
4. **Quality Assessment:** Rate the overall image quality based on focus, lighting, framing, and clarity.
5. **Guidance:** If there are issues, provide 1-2 sentences of child-friendly advice to help take a better photo. Be specific to what you see.

**Tone:** Helpful, slightly inquisitive, and deductive.

Provide your response in JSON format:

{
  "comprehensive_explanation": "Hmm, I see... Looking closely at this image, I notice... This looks like it could be... The image quality appears to be...",
  "image_quality": "GOOD|MODERATE|POOR",
  "quality_issues": ["issue1", "issue2"],
  "detected_objects": ["Object1", "Object2"],
  "guidance": "Child-friendly advice based on the actual issues you see. Empty string if no issues."
}

Notes:
- quality_issues should be an empty array [] if there are no issues
- detected_objects: list ACTUAL object names you can identify (e.g., "Building", "Hand", "Car")
- NEVER use descriptions in detected_objects (e.g., "red object", "blurry thing")
- guidance should address the MAIN issue you identified, not all issues
- For focus/blur problems: suggest moving camera closer or further (don't say "focus")
- Keep guidance brief and friendly
"""

        try:
            response = self._client.models.generate_content(
                model=self._config["model_name"],
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                    response_schema=ImageAnalysis
                )
            )

            if response.parsed:
                result = response.parsed
                # Set recommendation based on quality and objects
                result = self._set_recommendation(result)
                print(f"Stage 1 Analysis: {result.comprehensive_explanation}...")
                print(f"Quality: {result.image_quality}, Issues: {result.quality_issues}, Objects: {result.detected_objects}")
                print(f"Recommendation: {result.recommendation}")
                return result
            else:
                return ImageAnalysis(
                    comprehensive_explanation="Hmm, I'm having trouble analyzing this image clearly.",
                    image_quality="POOR",
                    quality_issues=["analysis_failed"],
                    detected_objects=[],
                    guidance="Try taking another photo!",
                    confidence_level="LOW",
                    recommendation="GUIDE"
                )

        except Exception as e:
            print(f"Error in analyze_image: {e}")
            return ImageAnalysis(
                comprehensive_explanation="Something went wrong while analyzing this image.",
                image_quality="POOR",
                quality_issues=["error"],
                detected_objects=[],
                guidance="Try taking another photo!",
                confidence_level="LOW",
                recommendation="GUIDE"
            )

    def _set_recommendation(self, analysis: ImageAnalysis) -> ImageAnalysis:
        """Set recommendation and confidence based on analysis results."""
        # Determine confidence based on quality
        if analysis.image_quality == "GOOD":
            analysis.confidence_level = "HIGH"
        elif analysis.image_quality == "MODERATE":
            analysis.confidence_level = "MEDIUM"
        else:
            analysis.confidence_level = "LOW"

        # Determine recommendation
        object_count = len(analysis.detected_objects)
        if analysis.image_quality == "POOR":
            analysis.recommendation = "GUIDE"
        elif object_count >= 2:
            analysis.recommendation = "MULTI_SELECT"
        elif object_count == 1:
            analysis.recommendation = "CLASSIFY"
        else:
            analysis.recommendation = "GUIDE"

        return analysis

    def recognize_object(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Union[RecognitionResult, MultiObjectResult, DiagnosticResult]:
        """
        Main entry point: analyze image and process based on recommendation.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image

        Returns:
            RecognitionResult, MultiObjectResult, or DiagnosticResult based on analysis
        """
        # Analyze the image (includes recommendation)
        analysis = self.analyze_image(image_bytes, mime_type)

        # Process based on recommendation
        return self._process_analysis(image_bytes, mime_type, analysis)

    def _process_analysis(self, image_bytes: bytes, mime_type: str, analysis: ImageAnalysis) -> Union[RecognitionResult, MultiObjectResult, DiagnosticResult]:
        """
        Process the analysis and return appropriate result.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image
            analysis: The analysis result with recommendation

        Returns:
            Appropriate result type based on recommendation
        """
        if analysis.recommendation == "GUIDE":
            return self._generate_guidance(analysis)

        elif analysis.recommendation == "MULTI_SELECT":
            return self._detect_objects_with_context(image_bytes, mime_type, analysis)

        else:  # CLASSIFY
            return self._classify_with_context(image_bytes, mime_type, analysis)

    def _generate_guidance(self, analysis: ImageAnalysis) -> DiagnosticResult:
        """
        Build DiagnosticResult using guidance from Stage 1 analysis.

        Args:
            analysis: The analysis result (includes guidance and detected_objects)

        Returns:
            DiagnosticResult with photo-specific friendly guidance
        """
        main_issue = analysis.quality_issues[0].upper() if analysis.quality_issues else "UNCLEAR"
        guesses = analysis.detected_objects[:3] if analysis.detected_objects else []
        friendly_message = analysis.guidance if analysis.guidance else "Try taking another photo!"

        print(f"Guidance from Stage 1: {friendly_message}")

        return DiagnosticResult(
            comprehensive_explanation=analysis.comprehensive_explanation,
            issue=main_issue,
            friendly_message=friendly_message,
            guesses=guesses
        )

    def _classify_with_context(self, image_bytes: bytes, mime_type: str, analysis: ImageAnalysis) -> RecognitionResult:
        """
        Classify a single object using context from the analysis.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image
            analysis: The analysis result

        Returns:
            RecognitionResult with object identification
        """
        context = f"Based on my initial analysis: {analysis.comprehensive_explanation}\n\nThe main object appears to be: {', '.join(analysis.detected_objects)}."

        prompt = f"""You are helping a child identify an object in a photo.

Context from initial analysis:
{context}

Now provide a final, confident identification with a fun, child-friendly description.

Rules:
- Focus on the MAIN object
- Give ONE clear answer
- Provide a confidence score from 0.0 to 1.0
- Use simple, child-friendly language (avoid technical terms)
- Make the description fun and educational!
- object_name must be the ACTUAL name of the object (e.g., "Banana", "Basketball", "Cat")
- NEVER use color + shape descriptions (e.g., "green object", "red sphere", "pink thing")
- If unsure, make your best guess at what the object actually IS

Output JSON format:
{{
  "object_name": "actual name of the object",
  "confidence": 0.95,
  "description": "A fun, simple sentence describing the object for a child"
}}

Examples of good object_name: "Banana", "Basketball", "Cat", "Book", "Cup"
Examples of BAD object_name: "yellow object", "round thing", "furry animal"

Examples of good descriptions:
- "A yummy yellow fruit that monkeys love!"
- "A round toy that you can bounce and play with!"
- "A fluffy friend that says meow!"
"""

        try:
            response = self._client.models.generate_content(
                model=self._config["model_name"],
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=RecognitionResult
                )
            )

            if response.parsed:
                result = response.parsed
                print(f"Classification result: {result.object_name} (confidence: {result.confidence})")
                return result
            else:
                return RecognitionResult(
                    object_name=analysis.detected_objects[0] if analysis.detected_objects else "Unknown",
                    confidence=0.5,
                    description="I found something interesting!"
                )

        except Exception as e:
            print(f"Error in _classify_with_context: {e}")
            return RecognitionResult(
                object_name="Unknown",
                confidence=0.0,
                description="Something went wrong! Let's try again."
            )

    def _detect_objects_with_context(self, image_bytes: bytes, mime_type: str, analysis: ImageAnalysis) -> MultiObjectResult:
        """
        Detect multiple objects with bounding boxes using context from analysis.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image
            analysis: The analysis result

        Returns:
            MultiObjectResult with objects and bounding boxes
        """
        object_count = len(analysis.detected_objects)
        context = f"Based on my initial analysis: {analysis.comprehensive_explanation}\n\nI detected {object_count} objects: {', '.join(analysis.detected_objects)}."

        prompt = f"""You are helping a child identify objects in a photo. There are multiple things visible.

Context from initial analysis:
{context}

Now provide details for each object including bounding boxes so we can show the child where each one is.

Rules:
- List 2-4 distinct objects that a child would point to
- Use simple, child-friendly names and descriptions
- Provide bounding boxes in [ymin, xmin, ymax, xmax] format, normalized to 0-1000 scale
- Make descriptions fun and educational!
- object_name must be the ACTUAL name of each object (e.g., "Apple", "Cup", "Book")
- NEVER use color + shape descriptions (e.g., "red object", "blue container", "round thing")
- If unsure, make your best guess at what the object actually IS

Output JSON format:
{{
  "objects": [
    {{
      "object_name": "Apple",
      "confidence": 0.9,
      "description": "A yummy red fruit that keeps the doctor away!",
      "box_2d": [ymin, xmin, ymax, xmax]
    }}
  ],
  "message": "I see a few things here! Which one do you want to know about?"
}}

Examples of good object_name: "Apple", "Cup", "Book", "Phone", "Pen"
Examples of BAD object_name: "red object", "white container", "rectangular thing"

IMPORTANT about box_2d:
- Format is [ymin, xmin, ymax, xmax] where each value is 0-1000
- 0 = top/left edge, 1000 = bottom/right edge
- The box should tightly surround each object
"""

        try:
            response = self._client.models.generate_content(
                model=self._config["model_name"],
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            )

            if response.text:
                text = response.text.strip()
                data = json.loads(text)

                if "objects" in data and len(data["objects"]) >= 2:
                    result = MultiObjectResult(
                        objects=data["objects"],
                        message=data.get("message", "I see multiple things! Which one interests you?")
                    )
                    print(f"Multi-object result: {[obj['object_name'] for obj in data['objects']]}")
                    return result

            # Fallback: create result from analysis
            return self._create_multi_object_fallback(analysis)

        except Exception as e:
            print(f"Error in _detect_objects_with_context: {e}")
            return self._create_multi_object_fallback(analysis)

    def _create_multi_object_fallback(self, analysis: ImageAnalysis) -> MultiObjectResult:
        """Create a fallback MultiObjectResult from analysis data."""
        objects = []
        for i, obj_name in enumerate(analysis.detected_objects[:4]):
            objects.append({
                "object_name": obj_name,
                "confidence": 0.7 - (i * 0.1),
                "description": f"This looks like a {obj_name}!",
                "box_2d": [100 + i*200, 100 + i*200, 400 + i*200, 400 + i*200]  # Placeholder boxes
            })

        return MultiObjectResult(
            objects=objects,
            message="I see a few things! Which one do you want to know about?"
        )

    def describe_object(self, object_name: str) -> RecognitionResult:
        """
        Generate a child-friendly description for a given object name.

        Args:
            object_name: The name of the object to describe

        Returns:
            RecognitionResult with the object name and a fun description
        """
        prompt = f"""Generate a fun, child-friendly description for: {object_name}

Rules:
- Write 1-2 simple sentences a child would enjoy
- Make it educational and fun
- Use simple words

Output JSON:
{{
  "object_name": "{object_name}",
  "confidence": 1.0,
  "description": "A fun, simple description for a child"
}}
"""

        try:
            response = self._client.models.generate_content(
                model=self._config["model_name"],
                contents=[
                    types.Part.from_text(text=prompt),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=RecognitionResult
                )
            )

            if response.parsed:
                result = response.parsed
                print(f"Generated description for {object_name}: {result.description}")
                return result
            else:
                return RecognitionResult(
                    object_name=object_name,
                    confidence=1.0,
                    description=f"This is a {object_name}!"
                )

        except Exception as e:
            print(f"Error in describe_object: {e}")
            return RecognitionResult(
                object_name=object_name,
                confidence=1.0,
                description=f"This is a {object_name}!"
            )
