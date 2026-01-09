import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------
# Build DCE-Net Model
# -------------------------------
def build_dce_net():
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
    model = build_dce_net()
    dummy_input = tf.random.normal([1, 256, 256, 3])
    _ = model(dummy_input)
    model.load_weights(weights_path)
    print(f"✅ Model loaded from '{weights_path}'")
    return model

# -------------------------------
# Brightness Detection
# -------------------------------
def get_brightness(image):
    """Calculate average brightness (0-255)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return np.mean(gray)

def check_if_needs_enhancement(image):
    """Check brightness and contrast to decide if enhancement is needed"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    std_dev = np.std(gray)
    
    # Low std_dev means low contrast
    needs_enhancement = brightness < 80 or std_dev < 40
    return needs_enhancement, brightness, std_dev

# -------------------------------
# Zero-DCE Enhancement
# -------------------------------
def apply_zerodce(model, image):
    """Apply Zero-DCE enhancement"""
    original_size = image.shape[:2]
    
    # Resize for model
    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = tf.expand_dims(image_normalized, axis=0)
    
    # Get curve parameters
    curve_params = model(image_tensor, training=False)
    curve_params = tf.squeeze(curve_params, axis=0)
    
    # Apply curves iteratively
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
    
    # Resize back to original size
    enhanced = cv2.resize(enhanced, (original_size[1], original_size[0]))
    
    return enhanced

# -------------------------------
# Improved Post-Processing Filters
# -------------------------------
def adaptive_clahe(img_bgr, brightness):
    """Apply CLAHE with adaptive parameters based on brightness"""
    # Adjust clip limit based on brightness
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
    """Apply gentle contrast enhancement"""
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
    """Simple color balancing using gray world assumption"""
    result = img_bgr.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
    
    return result.astype(np.uint8)

# -------------------------------
# Complete Enhancement Pipeline
# -------------------------------
def enhance_image_pipeline(model, image_path):
    """
    Improved adaptive enhancement pipeline
    """
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Analyze image
    needs_enhancement, brightness, contrast = check_if_needs_enhancement(img_rgb)
    print(f"📊 Brightness: {brightness:.1f}, Contrast (std): {contrast:.1f}")
    
    if needs_enhancement:
        print("🔧 Applying Zero-DCE enhancement...")
        
        # Apply Zero-DCE
        zerodce_output = apply_zerodce(model, img_rgb)
        result_bgr = cv2.cvtColor(zerodce_output, cv2.COLOR_RGB2BGR)
        
        # Check brightness after Zero-DCE
        new_brightness = get_brightness(zerodce_output)
        print(f"📊 After Zero-DCE brightness: {new_brightness:.1f}")
        
        # Apply gentle post-processing
        result_bgr = adaptive_clahe(result_bgr, new_brightness)
        result_bgr = gentle_contrast_enhancement(result_bgr, new_brightness)
        
        # Color balance if needed
        if new_brightness > 100:
            result_bgr = color_balance(result_bgr)
        
        # Gentle denoising
        result_bgr = cv2.fastNlMeansDenoisingColored(
            result_bgr, None, h=6, hColor=6, 
            templateWindowSize=7, searchWindowSize=21
        )
        
    else:
        print("✨ Image quality is good, applying minimal enhancement...")
        
        # Only apply very gentle adjustments
        result_bgr = adaptive_clahe(img_bgr, brightness)
        result_bgr = gentle_contrast_enhancement(result_bgr, brightness)
        result_bgr = cv2.fastNlMeansDenoisingColored(
            result_bgr, None, h=5, hColor=5,
            templateWindowSize=7, searchWindowSize=21
        )
    
    # Display results
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    final_brightness = get_brightness(result_rgb)
    
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original\nBrightness: {brightness:.0f}, Contrast: {contrast:.0f}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_rgb)
    plt.title(f"Enhanced\nBrightness: {final_brightness:.0f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Enhancement complete!")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Configuration
    WEIGHTS_PATH = "zero_dce_model_weights.h5"  # Change this to your weights file path
    IMAGE_PATH = r"E:\CLEAR-IMAGE\Zero-DCE\test_images\WhatsApp Image 2025-11-07 at 13.42.33_d7ed7396.jpg" # ⚠️ CHANGE THIS TO YOUR IMAGE PATH
    
    # Load model
    model = load_trained_model(WEIGHTS_PATH)
    
    # Run enhancement
    enhance_image_pipeline(model, IMAGE_PATH)