import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Import required TensorFlow components
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    # Ensure we're using the correct TensorFlow and Keras versions
    tf.keras.backend.clear_session()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print(f"Error importing TensorFlow: {e}")
    tf = None

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'user_model', 'keras_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'user_model', 'labels.txt')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'replace-me-with-a-random-secret'

# Globals
model = None
labels = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_labels(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def safe_load_model(path):
    if tf is None:
        raise RuntimeError('TensorFlow not available. Please install tensorflow.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found at {path}')
    try:
        # Load with custom_objects to handle special layers
        custom_objects = {
            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D
        }
        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print(f"Model loaded successfully. Input shape: {model.input_shape}")
        return model
    except Exception as e:
        raise RuntimeError(f'Error loading model: {str(e)}')


def preprocess_image(image: Image.Image, target_shape):
    """
    Resize and scale PIL image to fit model input.
    target_shape is a tuple like (height, width, channels) or (channels, height, width)
    """
    # Ensure RGB if 3 channels required, else convert to L for single channel
    if len(target_shape) == 3:
        # Determine channels position: assume (H,W,C)
        if target_shape[2] == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        target_h, target_w = target_shape[0], target_shape[1]
    else:
        # Fallback - try to infer H,W from shape
        # target_shape could be (None, H, W, C) trimmed earlier; handle common case
        target_h, target_w = target_shape[0], target_shape[1]
        image = image.convert('RGB')

    image = image.resize((target_w, target_h))
    arr = np.asarray(image).astype('float32') / 255.0
    # If grayscale, arr may be (H,W); make it (H,W,1)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    # Ensure shape (H,W,C)
    if arr.shape[-1] == 1 and (len(target_shape) == 3 and target_shape[2] == 3):
        # duplicate channels
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def model_input_shape(model):
    # model.input_shape often contains batch dim as None at position 0
    try:
        shape = model.input_shape
        # shape may be tuple like (None, H, W, C) or (None, C, H, W)
        if isinstance(shape, tuple):
            if len(shape) == 4:
                return tuple([s for s in shape[1:]])
            elif len(shape) == 3:
                return tuple([s for s in shape])
    except Exception:
        pass
    # fallback
    return (224, 224, 3)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model, labels
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load model and labels if not loaded
        try:
            if model is None:
                model = safe_load_model(MODEL_PATH)
            if labels is None:
                labels = load_labels(LABELS_PATH)
        except Exception as e:
            return render_template('index.html', error=str(e))

        # Open and preprocess image
        try:
            img = Image.open(filepath)
        except Exception as e:
            return render_template('index.html', error='Could not open image: ' + str(e))

        target_shape = model_input_shape(model)
        arr = preprocess_image(img, target_shape)

        # Adjust channels order if model expects channels_first
        # If model.input_shape after trimming is (C,H,W) detect by looking at first element being 1 or 3 and uncommon H/W values
        if len(target_shape) == 3 and target_shape[0] in (1, 3) and target_shape[0] < 5:
            # channels first expected
            arr = np.transpose(arr, (2, 0, 1))

        x = np.expand_dims(arr, axis=0)

        # Predict
        preds = model.predict(x)

        # Interpret predictions
        try:
            if preds.ndim == 2 and preds.shape[1] == 1:
                # binary probability
                prob = float(preds[0, 0])
                # if labels available, map: label[1] positive class
                if labels and len(labels) >= 2:
                    if prob >= 0.5:
                        pred_label = labels[1]
                    else:
                        pred_label = labels[0]
                else:
                    pred_label = 'Positive' if prob >= 0.5 else 'Negative'
                confidence = prob if prob >= 0.0 else 0.0
            else:
                # multiclass
                probs = np.squeeze(preds)
                if probs.ndim == 0:
                    # single value
                    idx = 0
                    confidence = float(probs)
                    pred_label = labels[0] if labels else 'Class 0'
                else:
                    idx = int(np.argmax(probs))
                    confidence = float(probs[idx])
                    pred_label = labels[idx] if labels and idx < len(labels) else f'Class {idx}'
        except Exception as e:
            return render_template('index.html', error='Error interpreting model output: ' + str(e))

        # Convert confidence to percent
        confidence_pct = round(confidence * 100, 2)

        return render_template('index.html', filename=filename, label=pred_label, confidence=confidence_pct)
    else:
        flash('File type not allowed')
        return redirect(request.url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # When run directly, do not force model load if tensorflow not installed; let the endpoint handle it and show errors.
    app.run(host='127.0.0.1', port=5000, debug=True)
