// Application State
const AppState = {
    currentScreen: 'welcome',
    capturedImage: null,
    stream: null,
    imageDataUrl: null
};

// DOM Elements
const screens = {
    welcome: document.getElementById('welcome-screen'),
    camera: document.getElementById('camera-screen'),
    preview: document.getElementById('preview-screen'),
    loading: document.getElementById('loading-screen'),
    result: document.getElementById('result-screen'),
    diagnostic: document.getElementById('diagnostic-screen'),
    selection: document.getElementById('selection-screen'),
    error: document.getElementById('error-screen')
};

const elements = {
    cameraBtn: document.getElementById('camera-btn'),
    uploadBtn: document.getElementById('upload-btn'),
    captureBtn: document.getElementById('capture-btn'),
    backFromCamera: document.getElementById('back-from-camera'),
    analyzeBtn: document.getElementById('analyze-btn'),
    retakeBtn: document.getElementById('retake-btn'),
    tryAgainBtn: document.getElementById('try-again-btn'),
    retakeDiagnosticBtn: document.getElementById('retake-diagnostic-btn'),
    retakeSelectionBtn: document.getElementById('retake-selection-btn'),
    backFromError: document.getElementById('back-from-error'),
    fileInput: document.getElementById('file-input'),
    video: document.getElementById('video'),
    previewImage: document.getElementById('preview-image'),
    resultImage: document.getElementById('result-image')
};

// Screen Navigation
function showScreen(screenName) {
    // Hide all screens
    Object.values(screens).forEach(screen => {
        screen.classList.remove('active');
    });

    // Show target screen
    if (screens[screenName]) {
        screens[screenName].classList.add('active');
        AppState.currentScreen = screenName;
    }
}

// Camera Functions
async function initCamera() {
    try {
        AppState.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',  // Use back camera on mobile
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        elements.video.srcObject = AppState.stream;
        showScreen('camera');
    } catch (error) {
        console.error('Camera error:', error);

        let errorMessage = 'Can\'t use camera! Let\'s upload a photo instead.';
        if (error.name === 'NotAllowedError') {
            errorMessage = 'We need camera permission! Check your browser settings, or upload a photo instead.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'No camera found! Let\'s upload a photo instead.';
        } else if (error.name === 'NotReadableError') {
            errorMessage = 'Camera is busy! Close other apps using it, or upload a photo.';
        }

        showError(errorMessage);
    }
}

function stopCamera() {
    if (AppState.stream) {
        AppState.stream.getTracks().forEach(track => track.stop());
        AppState.stream = null;
        elements.video.srcObject = null;
    }
}

function capturePhoto() {
    // Create canvas to capture video frame
    const canvas = document.createElement('canvas');
    canvas.width = elements.video.videoWidth;
    canvas.height = elements.video.videoHeight;

    const context = canvas.getContext('2d');
    context.drawImage(elements.video, 0, 0);

    // Convert to data URL
    AppState.imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);

    // Stop camera
    stopCamera();

    // Show preview
    elements.previewImage.src = AppState.imageDataUrl;
    showScreen('preview');
}

// File Upload Functions
function handleFileUpload() {
    elements.fileInput.click();
}

function processFile(file) {
    if (!file) return;

    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('Photo too big! Try a smaller one (less than 10MB).');
        return;
    }

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('This isn\'t a photo! Please choose a .jpg or .png file.');
        return;
    }

    // Read file
    const reader = new FileReader();
    reader.onload = (e) => {
        AppState.imageDataUrl = e.target.result;
        elements.previewImage.src = AppState.imageDataUrl;
        showScreen('preview');
    };
    reader.onerror = () => {
        showError('Could not read the file! Try another photo.');
    };
    reader.readAsDataURL(file);
}

// API Communication
async function analyzeImage() {
    if (!AppState.imageDataUrl) {
        showError('No image to analyze! Please take or upload a photo.');
        return;
    }

    showScreen('loading');

    try {
        const response = await fetch('/api/recognize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: AppState.imageDataUrl
            }),
            timeout: 120000  // 2 minute timeout
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Something went wrong!');
        }

        const data = await response.json();

        if (data.success) {
            if (data.result) {
                // High confidence - show result
                displaySuccess(data.result);
            } else if (data.multi_object) {
                // Multiple objects detected - show selection
                displayMultiObject(data.multi_object);
            } else if (data.diagnostic) {
                // Low confidence - show diagnostic
                displayDiagnostic(data.diagnostic);
            } else {
                throw new Error('Unexpected response format');
            }
        } else {
            throw new Error(data.error || 'Something went wrong!');
        }

    } catch (error) {
        console.error('API error:', error);

        let errorMessage = 'Something went wrong! Let\'s try again.';
        if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Can\'t connect! Check your internet connection.';
        } else if (error.message) {
            errorMessage = error.message;
        }

        showError(errorMessage);
    }
}

// Display Results
function displaySuccess(result) {
    const objectName = document.getElementById('object-name');
    const objectDescription = document.getElementById('object-description');
    const resultImage = document.getElementById('result-image');

    objectName.textContent = `It's a ${result.object_name.toUpperCase()}! 🎉`;
    objectDescription.textContent = result.description;
    resultImage.src = AppState.imageDataUrl;

    showScreen('result');
}

function displayDiagnostic(diagnostic) {
    const diagnosticMessage = document.getElementById('diagnostic-message');
    const guessesSection = document.getElementById('guesses-section');
    const guessesButtons = document.getElementById('guesses-buttons');

    diagnosticMessage.textContent = diagnostic.friendly_message;

    // Clear previous guesses
    guessesButtons.innerHTML = '';

    // Hide guesses section if no guesses, show if there are guesses
    if (!diagnostic.guesses || diagnostic.guesses.length === 0) {
        guessesSection.style.display = 'none';
    } else {
        guessesSection.style.display = 'block';

        // Create guess buttons
        diagnostic.guesses.forEach((guess, index) => {
            const button = document.createElement('button');
            button.className = 'guess-button';
            button.textContent = guess;

            // Add confidence indicator if available
            if (diagnostic.confidence_of_guesses && diagnostic.confidence_of_guesses[index]) {
                const confidence = Math.round(diagnostic.confidence_of_guesses[index] * 100);
                button.textContent += ` (${confidence}%)`;
            }

            button.onclick = () => handleGuessSelection(guess);
            guessesButtons.appendChild(button);
        });
    }

    showScreen('diagnostic');
}

async function handleGuessSelection(guess) {
    // Show loading state
    showScreen('loading');

    try {
        const response = await fetch('/api/describe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ object_name: guess })
        });

        const data = await response.json();

        if (data.success && data.result) {
            displaySuccess(data.result);
        } else {
            // Fallback if API fails
            displaySuccess({
                object_name: guess,
                confidence: 1.0,
                description: `This is a ${guess}!`
            });
        }
    } catch (error) {
        console.error('Error getting description:', error);
        // Fallback on error
        displaySuccess({
            object_name: guess,
            confidence: 1.0,
            description: `This is a ${guess}!`
        });
    }
}

// Colors for buttons
const BUTTON_COLORS = ['#A855F7', '#3B82F6', '#10B981', '#F59E0B'];

function displayMultiObject(multiObject) {
    const selectionMessage = document.getElementById('selection-message');
    const objectButtons = document.getElementById('object-buttons');
    const selectionImage = document.getElementById('selection-image');

    selectionMessage.textContent = multiObject.message;

    // Display the image
    selectionImage.src = AppState.imageDataUrl;

    // Clear previous buttons
    objectButtons.innerHTML = '';

    // Create color-coded buttons for each detected object
    multiObject.objects.forEach((obj, index) => {
        const color = BUTTON_COLORS[index % BUTTON_COLORS.length];
        const button = document.createElement('button');
        button.className = 'object-button';
        button.textContent = obj.object_name;
        button.style.borderColor = color;
        button.style.color = color;
        button.dataset.color = color;

        // Handle hover effects with dynamic color
        button.addEventListener('mouseenter', () => {
            button.style.backgroundColor = color;
            button.style.color = '#FFFFFF';
        });
        button.addEventListener('mouseleave', () => {
            button.style.backgroundColor = '#FFFFFF';
            button.style.color = color;
        });

        button.onclick = () => handleObjectSelection(obj);
        objectButtons.appendChild(button);
    });

    showScreen('selection');
}

function handleObjectSelection(obj) {
    // User selected an object - display its details
    const result = {
        object_name: obj.object_name,
        confidence: obj.confidence,
        description: obj.description
    };
    displaySuccess(result);
}

// Error Handling
function showError(message) {
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    showScreen('error');
}

// Event Listeners
elements.cameraBtn.addEventListener('click', initCamera);
elements.uploadBtn.addEventListener('click', handleFileUpload);
elements.captureBtn.addEventListener('click', capturePhoto);
elements.backFromCamera.addEventListener('click', () => {
    stopCamera();
    showScreen('welcome');
});

elements.analyzeBtn.addEventListener('click', analyzeImage);
elements.retakeBtn.addEventListener('click', () => {
    AppState.imageDataUrl = null;
    showScreen('welcome');
});

elements.tryAgainBtn.addEventListener('click', () => {
    AppState.imageDataUrl = null;
    showScreen('welcome');
});

elements.retakeDiagnosticBtn.addEventListener('click', () => {
    AppState.imageDataUrl = null;
    showScreen('welcome');
});

elements.retakeSelectionBtn.addEventListener('click', () => {
    AppState.imageDataUrl = null;
    showScreen('welcome');
});

elements.backFromError.addEventListener('click', () => {
    showScreen('welcome');
});

elements.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    processFile(file);
    // Reset file input so same file can be selected again
    e.target.value = '';
});

// Keyboard shortcuts for accessibility
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (AppState.currentScreen === 'camera') {
            stopCamera();
            showScreen('welcome');
        }
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
});

// Initialize app
console.log('Object Detective ready! 🔍');
