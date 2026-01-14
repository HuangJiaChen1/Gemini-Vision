from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from services.gemini_service import GeminiService
from services.image_processor import ImageProcessor
from models.response_models import ApiResponse, RecognitionResult, MultiObjectResult, DiagnosticResult

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Initialize services
gemini_service = GeminiService()
image_processor = ImageProcessor()


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Object Detective is ready!'
    })


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    Main recognition endpoint
    Accepts: multipart/form-data with 'image' file OR JSON with 'image' base64
    Returns: ApiResponse with either recognition result or diagnostic
    """
    try:
        # Get image data from request
        image_bytes = None
        mime_type = "image/jpeg"

        # Check if it's multipart form data (file upload)
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No image selected! Please choose a photo.'
                }), 400

            image_bytes = file.read()

        # Check if it's JSON with base64 (camera capture)
        elif request.is_json and 'image' in request.json:
            base64_data = request.json['image']
            try:
                image_bytes = image_processor.process_base64(base64_data)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Could not read the image! Try again.'
                }), 400

        else:
            return jsonify({
                'success': False,
                'error': 'No image provided! Please send a photo.'
            }), 400

        # Validate image
        is_valid, error_message = image_processor.validate_image(image_bytes)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_message
            }), 400

        # Resize/compress if needed
        image_bytes = image_processor.resize_image(image_bytes)
        mime_type = image_processor.get_mime_type(image_bytes)

        print(f"Processing image: {len(image_bytes)} bytes, {mime_type}")

        # Two-stage analysis-first flow:
        # Stage 1: Analyze image (quality, objects, recommendation)
        # Stage 2: Process based on analysis (classify, multi-select, or guide)
        result = gemini_service.recognize_object(image_bytes, mime_type)

        # Build response based on result type
        if isinstance(result, MultiObjectResult):
            print(f"Multiple objects detected: {[obj.object_name for obj in result.objects]}")
            response = ApiResponse(
                success=True,
                multi_object=result
            )
        elif isinstance(result, DiagnosticResult):
            print(f"Guidance needed: {result.issue}")
            response = ApiResponse(
                success=True,
                diagnostic=result
            )
        else:  # RecognitionResult
            print(f"Object identified: {result.object_name} (confidence: {result.confidence})")
            response = ApiResponse(
                success=True,
                result=result
            )

        return jsonify(response.model_dump())

    except Exception as e:
        print(f"Error in /api/recognize: {e}")
        return jsonify({
            'success': False,
            'error': 'Something went wrong! Let\'s try again.'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Page not found!'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Something went wrong! Please try again.'
    }), 500


if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Run Flask app
    print("Starting Object Detective web app...")
    print("Open http://localhost:5000 in your browser!")
    app.run(debug=True, host='0.0.0.0', port=5000)
