import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
fname = 'cam_output.jpg'
SAVE_PATH = os.path.join(BASE_DIR, 'static', 'results', 'cam_output.jpg')

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static', 'results')
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

print("Loading Model...")
model = load_model(MODEL_PATH)

def get_gradcam(img_array, last_conv_layer_name='conv2d_5'):
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_output = None
        
        # عبور لایه به لایه برای پیدا کردن خروجی کانولوشن و خروجی نهایی
        for layer in model.layers:
            if layer == model.layers[-1] and hasattr(layer, 'activation'):
                continue 
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)
        
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.reduce_mean(conv_output, axis=-1)
        max_val = tf.math.reduce_max(heatmap)
    
    heatmap = heatmap / (max_val + 1e-10)
    return heatmap.numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_resized = img.resize((180, 180))
    img_array = img_to_array(img_resized)


    img_array -= np.mean(img_array, keepdims=True)
    img_array /= (np.std(img_array, keepdims=True) + 1e-6)
    
    img_array = img_array.astype('float32')
    img_input = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_input)
    confidence = float(prediction[0][0])
    
    print(f"\n🔍 DEBUG - Raw Prediction: {confidence}")

    is_pneumonia = confidence > 0.5
    label = "PNEUMONIA" if is_pneumonia else "NORMAL"

    response = {
        'label': label,
        'confidence': round(confidence * 100, 2) if is_pneumonia else round((1-confidence)*100, 2),
        'show_gradcam': False
    }

    if is_pneumonia:
        heatmap = get_gradcam(img_input, last_conv_layer_name='conv2d_5')
        
        img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        
        heatmap_resized = cv2.resize(heatmap, (180, 180))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        superimposed = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
        
        cv2.imwrite(SAVE_PATH, superimposed)
        print(f"✅ Grad-CAM saved. Heatmap Max: {np.max(heatmap)}")

        response['gradcam_url'] = url_for('static', filename='results/' + fname) + "?t=" + str(np.random.randint(1000))
        response['show_gradcam'] = True

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)