import json
from typing import Optional, Union
from google import genai
from google.genai import types
from models.response_models import RecognitionResult, DiagnosticResult, MultiObjectResult, ImageAnalysis, ImageDecision


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

**Tone:** Helpful, slightly inquisitive, and deductive.

Provide your response in JSON format. The "comprehensive_explanation" field must describe exactly what you see, written in the stream-of-thought style above. Be concise but thorough.

{
  "comprehensive_explanation": "Hmm, I see... Looking closely at this image, I notice... This looks like it could be... The image quality appears to be...",
  "image_quality": "GOOD|MODERATE|POOR",
  "quality_issues": ["issue1", "issue2"]
}

Notes:
- quality_issues should be an empty array [] if there are no issues
- Focus on OBSERVATION, not decisions about what to do next
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
                print(f"Stage 1 Analysis: {result.comprehensive_explanation}...")
                print(f"Quality: {result.image_quality}, Issues: {result.quality_issues}")
                return result
            else:
                return ImageAnalysis(
                    comprehensive_explanation="Hmm, I'm having trouble analyzing this image clearly.",
                    image_quality="POOR",
                    quality_issues=["analysis_failed"]
                )

        except Exception as e:
            print(f"Error in analyze_image: {e}")
            return ImageAnalysis(
                comprehensive_explanation="Something went wrong while analyzing this image.",
                image_quality="POOR",
                quality_issues=["error"]
            )

    def make_decision(self, analysis: ImageAnalysis) -> ImageDecision:
        """
        Stage 2: Make decisions based on the analysis.
        Interprets the analysis to determine object count, names, confidence, and next action.

        Args:
            analysis: The Stage 1 analysis result

        Returns:
            ImageDecision with object details and recommendation
        """
        prompt = f"""Based on this image analysis, make decisions about what to do next.

IMAGE ANALYSIS:
{analysis.comprehensive_explanation}

Image Quality: {analysis.image_quality}
Quality Issues: {', '.join(analysis.quality_issues) if analysis.quality_issues else 'None'}

YOUR TASK:
1. Count how many distinct objects are mentioned in the analysis
2. List the names of objects detected
3. Assess confidence level (HIGH if clear identification, MEDIUM if somewhat clear, LOW if uncertain)
4. Recommend next action:
   - CLASSIFY: One clear main object, good/moderate quality → identify it
   - MULTI_SELECT: Multiple distinct objects (2-4) mentioned → let user pick which one
   - GUIDE: Poor quality OR unclear OR quality issues present → help user take better photo

Output JSON:
{{
  "object_count": 1,
  "detected_objects": ["object name"],
  "confidence_level": "HIGH|MEDIUM|LOW",
  "recommendation": "CLASSIFY|MULTI_SELECT|GUIDE"
}}

Rules:
- If image_quality is POOR, recommendation should usually be GUIDE
- If multiple distinct objects are clearly mentioned, recommendation should be MULTI_SELECT
- If one clear object with good quality, recommendation should be CLASSIFY
- confidence_level reflects how certain the analysis is about object identification
"""

        try:
            response = self._client.models.generate_content(
                model=self._config["model_name"],
                contents=[
                    types.Part.from_text(text=prompt),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ImageDecision
                )
            )

            if response.parsed:
                result = response.parsed
                print(f"Stage 2 Decision: objects={result.detected_objects}, confidence={result.confidence_level}, recommendation={result.recommendation}")
                return result
            else:
                return ImageDecision(
                    object_count=0,
                    detected_objects=[],
                    confidence_level="LOW",
                    recommendation="GUIDE"
                )

        except Exception as e:
            print(f"Error in make_decision: {e}")
            return ImageDecision(
                object_count=0,
                detected_objects=[],
                confidence_level="LOW",
                recommendation="GUIDE"
            )

    def recognize_object(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Union[RecognitionResult, MultiObjectResult, DiagnosticResult]:
        """
        Main entry point using three-stage architecture.

        Stage 1: Analyze the image (observation only)
        Stage 2: Make decision based on analysis
        Stage 3: Process based on decision (classify, multi-select, or guide)

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image

        Returns:
            RecognitionResult, MultiObjectResult, or DiagnosticResult based on decision
        """
        # Stage 1: Pure observation - analyze the image
        analysis = self.analyze_image(image_bytes, mime_type)

        # Stage 2: Make decision based on analysis
        decision = self.make_decision(analysis)

        # Stage 3: Process based on decision
        return self._process_with_decision(image_bytes, mime_type, analysis, decision)

    def _process_with_decision(self, image_bytes: bytes, mime_type: str, analysis: ImageAnalysis, decision: ImageDecision) -> Union[RecognitionResult, MultiObjectResult, DiagnosticResult]:
        """
        Stage 3: Execute the appropriate action based on decision.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image
            analysis: The Stage 1 analysis result
            decision: The Stage 2 decision result

        Returns:
            Appropriate result type based on recommendation
        """
        if decision.recommendation == "GUIDE":
            # Image has quality issues - generate guidance for the child
            return self._generate_guidance(analysis, decision)

        elif decision.recommendation == "MULTI_SELECT":
            # Multiple objects detected - get details with bounding boxes
            return self._detect_objects_with_context(image_bytes, mime_type, analysis, decision)

        else:  # CLASSIFY
            # Single clear object - classify it with context from analysis
            return self._classify_with_context(image_bytes, mime_type, analysis, decision)

    def _generate_guidance(self, analysis: ImageAnalysis, decision: ImageDecision) -> DiagnosticResult:
        """
        Generate child-friendly guidance based on image quality issues.

        Args:
            analysis: The Stage 1 analysis result
            decision: The Stage 2 decision result

        Returns:
            DiagnosticResult with friendly guidance
        """
        # Map quality issues to friendly messages
        issue_messages = {
            "blur": "The photo is a bit blurry! Try holding the camera very still.",
            "dark": "It's too dark to see! Try moving to a brighter spot.",
            "bright": "It's too bright! Try moving away from the light.",
            "cropped": "Part of the object is cut off! Try stepping back a little.",
            "obstruction": "Something is blocking the view! Try moving it out of the way.",
            "too_close": "You're too close! Try stepping back so I can see the whole thing.",
            "too_far": "You're too far away! Try getting closer so I can see better.",
            "multiple_unclear": "I see a few things but I'm not sure which one you want! Try pointing at just one thing.",
        }

        # Determine the main issue and friendly message
        main_issue = analysis.quality_issues[0] if analysis.quality_issues else "UNCLEAR"
        friendly_message = issue_messages.get(
            main_issue.lower(),
            "Hmm, I'm having trouble seeing clearly! Try taking another photo."
        )

        # Use detected objects from decision as guesses
        guesses = decision.detected_objects[:3] if decision.detected_objects else ["something", "an object", "a thing"]
        confidence_scores = [0.4, 0.3, 0.2][:len(guesses)]

        return DiagnosticResult(
            comprehensive_explanation=analysis.comprehensive_explanation,
            issue=main_issue.upper() if main_issue else "UNCLEAR",
            friendly_message=friendly_message,
            guesses=guesses,
            confidence_of_guesses=confidence_scores
        )

    def _classify_with_context(self, image_bytes: bytes, mime_type: str, analysis: ImageAnalysis, decision: ImageDecision) -> RecognitionResult:
        """
        Classify a single object using context from the analysis.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image
            analysis: The Stage 1 analysis result
            decision: The Stage 2 decision result

        Returns:
            RecognitionResult with object identification
        """
        # Use the analysis and decision context in the prompt
        context = f"Based on my initial analysis: {analysis.comprehensive_explanation}\n\nThe main object appears to be: {', '.join(decision.detected_objects)}."

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

Output JSON format:
{{
  "object_name": "friendly name of the object",
  "confidence": 0.95,
  "description": "A fun, simple sentence describing the object for a child"
}}

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
                # Use decision as fallback
                return RecognitionResult(
                    object_name=decision.detected_objects[0] if decision.detected_objects else "Unknown",
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

    def _detect_objects_with_context(self, image_bytes: bytes, mime_type: str, analysis: ImageAnalysis, decision: ImageDecision) -> MultiObjectResult:
        """
        Detect multiple objects with bounding boxes using context from analysis.

        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image
            analysis: The Stage 1 analysis result
            decision: The Stage 2 decision result

        Returns:
            MultiObjectResult with objects and bounding boxes
        """
        # Use the analysis and decision context in the prompt
        context = f"Based on my initial analysis: {analysis.comprehensive_explanation}\n\nI detected {decision.object_count} objects: {', '.join(decision.detected_objects)}."

        prompt = f"""You are helping a child identify objects in a photo. There are multiple things visible.

Context from initial analysis:
{context}

Now provide details for each object including bounding boxes so we can show the child where each one is.

Rules:
- List 2-4 distinct objects that a child would point to
- Use simple, child-friendly names and descriptions
- Provide bounding boxes in [ymin, xmin, ymax, xmax] format, normalized to 0-1000 scale
- Make descriptions fun and educational!

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

            # Fallback: create result from decision
            return self._create_multi_object_fallback(decision)

        except Exception as e:
            print(f"Error in _detect_objects_with_context: {e}")
            return self._create_multi_object_fallback(decision)

    def _create_multi_object_fallback(self, decision: ImageDecision) -> MultiObjectResult:
        """Create a fallback MultiObjectResult from decision data."""
        objects = []
        for i, obj_name in enumerate(decision.detected_objects[:4]):
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
