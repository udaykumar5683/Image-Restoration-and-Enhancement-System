import os 
import cv2 
import numpy as np 
from flask import Flask, request, jsonify, render_template, send_from_directory 
import sys

# -------------------------------
# Build DCE-Net Model 
# -------------------------------
def build_dce_net(): 
    from tensorflow import keras 
    from tensorflow.keras import layers 
    input_img = keras.Input(shape=[None, None, 3]) 
    conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img) 
    conv2 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv1) 
    conv3 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv2) 
    conv4 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv3) 
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3]) 
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
    try:
        import tensorflow as tf
        model = build_dce_net() 
        dummy_input = tf.random.normal([1, 256, 256, 3]) 
        _ = model(dummy_input) 
        model.load_weights(weights_path) 
        print(f"✅ Model loaded from '{weights_path}'") 
        return model 
    except Exception:
        return None 

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
    import tensorflow as tf
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

def gentle_contrast_enhancement(img, brightness):  #Gaussian Blur Unsharp Masking
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
# Overexposure Correction imports (Pix2PixHD) - used when image is overly bright
OVEREXPOSURE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Overexposure Correction'))
if OVEREXPOSURE_DIR not in sys.path:
    sys.path.append(OVEREXPOSURE_DIR)
try:
    from options.test_options import TestOptions
    from models.models import create_model
    from data.base_dataset import get_params, get_transform
    from util.util import tensor2im
    HAS_OVEREXPOSURE = True
except Exception:
    HAS_OVEREXPOSURE = False


# Function to apply overexposure correction using Pix2PixHD model
def apply_overexposure_correction(image_rgb):
    if not HAS_OVEREXPOSURE:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # lazy import of PIL
    from PIL import Image
    if 'overexp_opt' not in globals() or globals().get('overexp_model') is None:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    pil = Image.fromarray(image_rgb)
    params = get_params(overexp_opt, pil.size)
    transform_A = get_transform(overexp_opt, params)
    A_tensor = transform_A(pil).unsqueeze(0)
    generated = overexp_model.inference(A_tensor, None)
    out_np = tensor2im(generated.data[0])
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr

# -------------------------------
# Flask App 
# -------------------------------
app = Flask(__name__) 

app.config['UPLOAD_FOLDER'] = 'static/uploads' 
app.config['RESULT_FOLDER'] = 'static/results' 

WEIGHTS_PATH = "zero_dce_model_weights.h5" 

try: 
    model = load_trained_model(WEIGHTS_PATH) 
except Exception: 
    model = None 

# Prepare Overexposure Correction model (if available)
overexp_model = None
overexp_opt = None
if HAS_OVEREXPOSURE:
    try:
        # Force parse to use CPU by overriding argv so that Pix2PixHD doesn't try to use GPU
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
    except Exception:
        overexp_model = None
        overexp_opt = None

BACKGROUND_DIR_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontendbackgroundimages')) 
BACKGROUND_DIR_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backgroundimages')) 
BEFORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'before')) 
AFTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'after')) 

def _build_pairs(limit=None): 
    try: 
        def normalize(name: str) -> str: 
            base, _ = os.path.splitext(name) 
            low = base.lower() 
            if low.endswith('before'): 
                return base[: -len('before')] 
            if low.endswith('after'): 
                return base[: -len('after')] 
            return base 

        before_map = {} 
        for f in os.listdir(BEFORE_DIR): 
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")): 
                key = normalize(f) 
                before_map[key] = f 

        after_map = {} 
        for f in os.listdir(AFTER_DIR): 
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")): 
                key = normalize(f) 
                after_map[key] = f 

        keys = sorted(list(set(before_map.keys()).intersection(after_map.keys()))) 
        if limit is not None: 
            keys = keys[:limit] 

        pairs = [{ 
            'name': k, 
            'before': f"/examples/before/{before_map[k]}", 
            'after': f"/examples/after/{after_map[k]}" 
        } for k in keys] 

        return pairs 

    except Exception: 
        return [] 

def get_sample_pairs(limit=8): 
    return _build_pairs(limit=limit) 

def list_images(dir_path): 
    try: 
        files = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))] 
        files.sort() 
        return files 
    except Exception: 
        return [] 

def get_all_pairs(): 
    return _build_pairs(limit=None) 

@app.route('/') 
def index(): 
    images = [] 
    try: 
        files = [] 
        for d in [BACKGROUND_DIR_1, BACKGROUND_DIR_2]: 
            if os.path.isdir(d): 
                files.extend([f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]) 
        files = sorted(list(dict.fromkeys(files))) 
        images = [f"/backgrounds/{f}" for f in files] 
    except Exception: 
        images = [] 

    sample_pairs = get_sample_pairs(limit=8) 
    pairs_all = get_all_pairs() 

    before_images = [f"/examples/before/{f}" for f in list_images(BEFORE_DIR)] 
    after_images = [f"/examples/after/{f}" for f in list_images(AFTER_DIR)] 

    return render_template('index.html', background_images=images, sample_pairs=sample_pairs, pairs_all=pairs_all, before_images=before_images, after_images=after_images) 

@app.route('/backgrounds/<path:filename>') 
def backgrounds(filename): 
    for d in [BACKGROUND_DIR_1, BACKGROUND_DIR_2]: 
        path = os.path.join(d, filename) 
        if os.path.isfile(path): 
            return send_from_directory(d, filename) 
    return send_from_directory(BACKGROUND_DIR_1, filename) 

@app.route('/examples/before/<path:filename>') 
def examples_before(filename): 
    return send_from_directory(BEFORE_DIR, filename) 

@app.route('/examples/after/<path:filename>') 
def examples_after(filename): 
    return send_from_directory(AFTER_DIR, filename) 

@app.route('/get-started') 
def get_started(): 
    images = [] 
    try: 
        files = [] 
        for d in [BACKGROUND_DIR_1, BACKGROUND_DIR_2]: 
            if os.path.isdir(d): 
                files.extend([f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]) 
        files = sorted(list(dict.fromkeys(files))) 
        images = [f"/backgrounds/{f}" for f in files] 
    except Exception: 
        images = [] 

    return render_template('get_started.html', background_images=images) 

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
        # Ensure upload/result folders exist before saving
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
        file.save(original_path) 

        # Image Enhancement Pipeline 
        img_bgr = cv2.imread(original_path) 
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 

        needs_enhancement, brightness, contrast = check_if_needs_enhancement(img_rgb) 

        # Decide route: Overexposed -> overexposure correction, else -> Zero-DCE/minimal
        OVEREXPOSED_THRESHOLD = 170.0
        if brightness >= OVEREXPOSED_THRESHOLD and HAS_OVEREXPOSURE and overexp_model is not None:
            result_bgr = apply_overexposure_correction(img_rgb)
            new_brightness = get_brightness(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            result_bgr = adaptive_clahe(result_bgr, new_brightness)
            result_bgr = gentle_contrast_enhancement(result_bgr, new_brightness)
            if new_brightness > 100:
                result_bgr = color_balance(result_bgr)
            result_bgr = cv2.fastNlMeansDenoisingColored(result_bgr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)

        elif needs_enhancement and model is not None: 
            zerodce_output = apply_zerodce(model, img_rgb) 
            result_bgr = cv2.cvtColor(zerodce_output, cv2.COLOR_RGB2BGR) 

            new_brightness = get_brightness(zerodce_output) 
            result_bgr = adaptive_clahe(result_bgr, new_brightness) 
            result_bgr = gentle_contrast_enhancement(result_bgr, new_brightness) 

            if new_brightness > 100: 
                result_bgr = color_balance(result_bgr) 

            result_bgr = cv2.fastNlMeansDenoisingColored( 
                result_bgr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21 
            ) 

        else: 
            result_bgr = adaptive_clahe(img_bgr, brightness) 
            result_bgr = gentle_contrast_enhancement(result_bgr, brightness) 
            result_bgr = cv2.fastNlMeansDenoisingColored( 
                result_bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21 
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
