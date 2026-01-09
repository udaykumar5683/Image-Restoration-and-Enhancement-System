import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow import keras
from tensorflow.keras import layers

# Overexposure Correction imports (Pix2PixHD)
OVEREXPOSURE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Overexposure Correction'))
if OVEREXPOSURE_DIR not in sys.path:
    sys.path.append(OVEREXPOSURE_DIR)
from options.test_options import TestOptions
from models.models import create_model # Builds the generator network used for overexposure correction.
from data.base_dataset import get_params, get_transform # Prepares image data for Pix2PixHD training.
from util.util import tensor2im # Converts a TensorFlow tensor to a NumPy array for image display.

# -------------------------------
# Build DCE-Net Model
# -------------------------------
def build_dce_net():
    input_img = keras.Input(shape=[None, None, 3]) # Input layer for the image (None, None, 3) for variable-size images.
    conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
    conv2 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv1)
    conv3 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv2)
    conv4 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    # it follows u-net architecture with skip connections
    conv5 = layers.Conv2D(32, 3, activation="relu", padding="same")(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, 3, activation="relu", padding="same")(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, 3, activation="tanh", padding="same")(int_con3)
    return keras.Model(inputs=input_img, outputs=x_r)

# -------------------------------
# Load Model Weights
# -------------------------------
def load_trained_model(weights_path='zero_dce_model_weights.h5'):
    model = build_dce_net()
    dummy_input = tf.random.normal([1, 256, 256, 3])
    _ = model(dummy_input)
    model.load_weights(weights_path)
    print(f"✅ Model loaded from '{weights_path}'")
    return model

# -------------------------------
# Image Enhancement Functions (from main.py)
# -------------------------------
def get_brightness(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return np.mean(gray)

def apply_zerodce(model, image):
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = tf.expand_dims(image_normalized, axis=0)
    
    curve_params = model(image_tensor, training=False)
    curve_params = tf.squeeze(curve_params, axis=0)
    
    def apply_curve(x, r):
        return x + r * (tf.square(x) - x)
    
    r1, r2, r3, r4, r5, r6, r7, r8 = tf.split(curve_params, num_or_size_splits=8, axis=-1)
    x = apply_curve(image_normalized, r1)
    x = apply_curve(x, r2)
    x = apply_curve(x, r3)
    x = apply_curve(x, r4)
    x = apply_curve(x, r5)
    x = apply_curve(x, r6)
    x = apply_curve(x, r7)
    enhanced = apply_curve(x, r8)
    
    enhanced = np.clip(enhanced.numpy(), 0, 1)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    enhanced = cv2.resize(enhanced, (original_size[1], original_size[0]))
    return enhanced

def adaptive_clahe(img_bgr, brightness):
    if brightness < 60:
        clip_limit = 2.0
    elif brightness < 100:
        clip_limit = 1.5
    else:
        clip_limit = 1.0
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L)
    lab_enhanced = cv2.merge([L_enhanced, A, B])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def gentle_contrast_enhancement(img, brightness):
    if brightness < 60:
        sigma, gain = 15, 0.8
    elif brightness < 100:
        sigma, gain = 12, 0.5
    else:
        sigma, gain = 10, 0.3
    
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    enhanced = cv2.addWeighted(img, 1 + gain, blur, -gain, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def color_balance(img_bgr):
    result = img_bgr.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
    
    return result.astype(np.uint8)

def check_if_needs_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    std_dev = np.std(gray)
    needs_enhancement = brightness < 80 or std_dev < 40
    return needs_enhancement, brightness, std_dev

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
WEIGHTS_PATH = "zero_dce_model_weights.h5"
model = load_trained_model(WEIGHTS_PATH)

# Prepare Overexposure Correction model (CPU by default)
# Force parse to use CPU by overriding argv
sys.argv = ['app.py', '--gpu_ids', '-1']
overexp_opt = TestOptions().parse(save=False)
overexp_opt.isTrain = False
overexp_opt.gpu_ids = []
overexp_opt.checkpoints_dir = os.path.join(OVEREXPOSURE_DIR, 'checkpoints')
overexp_opt.name = 'exposure_correction_experiment'
overexp_opt.which_epoch = 'latest'
overexp_opt.loadSize = 256
overexp_opt.fineSize = 256
overexp_opt.label_nc = 3
overexp_opt.output_nc = 3
overexp_opt.netG = 'global'
overexp_opt.ngf = 64
overexp_opt.n_downsample_global = 4
overexp_opt.n_blocks_global = 9
overexp_opt.batchSize = 1
overexp_opt.resize_or_crop = 'resize_same'
overexp_model = create_model(overexp_opt)

def apply_overexposure_correction(image_rgb):
    from PIL import Image
    pil = Image.fromarray(image_rgb)
    params = get_params(overexp_opt, pil.size)
    transform_A = get_transform(overexp_opt, params)
    A_tensor = transform_A(pil).unsqueeze(0)
    generated = overexp_model.inference(A_tensor, None)
    out_np = tensor2im(generated.data[0])
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        file.save(original_path)

        # Image Enhancement Pipeline
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

        img_bgr = cv2.imread(original_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        needs_enhancement, brightness, contrast = check_if_needs_enhancement(img_rgb)

        # Decide route: Overexposed → Overexposure Correction; else → Zero-DCE/minimal
        OVEREXPOSED_THRESHOLD = 170.0
        if brightness >= OVEREXPOSED_THRESHOLD:
            result_bgr = apply_overexposure_correction(img_rgb)
            new_brightness = get_brightness(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            result_bgr = adaptive_clahe(result_bgr, new_brightness)
            result_bgr = gentle_contrast_enhancement(result_bgr, new_brightness)
            if new_brightness > 100:
                result_bgr = color_balance(result_bgr)
            result_bgr = cv2.fastNlMeansDenoisingColored(
                result_bgr, None, h=6 , hColor=6,   #Strength of noise removal for the luminance (brightness) channel. Higher value = more noise removed Too high = image becomes blurry
                templateWindowSize=7, searchWindowSize=21 #Small patch size used to compare pixels. Area size used to search for similar pixels.
            )
        elif needs_enhancement:
            zerodce_output = apply_zerodce(model, img_rgb)
            result_bgr = cv2.cvtColor(zerodce_output, cv2.COLOR_RGB2BGR)
            new_brightness = get_brightness(zerodce_output)
            result_bgr = adaptive_clahe(result_bgr, new_brightness)
            result_bgr = gentle_contrast_enhancement(result_bgr, new_brightness)
            if new_brightness > 100:
                result_bgr = color_balance(result_bgr)
            result_bgr = cv2.fastNlMeansDenoisingColored(
                result_bgr, None, h=6, hColor=6,
                templateWindowSize=7, searchWindowSize=21
            )
        else:
            result_bgr = adaptive_clahe(img_bgr, brightness)
            result_bgr = gentle_contrast_enhancement(result_bgr, brightness)
            result_bgr = cv2.fastNlMeansDenoisingColored(
                result_bgr, None, h=5, hColor=5,
                templateWindowSize=7, searchWindowSize=21
            )

        cv2.imwrite(result_path, result_bgr)

        brightness_after = get_brightness(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

        return jsonify({
            'original': f'/static/uploads/{filename}',
            'enhanced': f'/static/results/{filename}',
            'brightness_before': float(brightness),
            'brightness_after': float(brightness_after)
        })

if __name__ == '__main__':
    app.run(debug=True)