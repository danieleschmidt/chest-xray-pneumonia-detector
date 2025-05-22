import os
import io
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuration (consider moving to a config file or env variables later)
# MODEL_PATH = os.environ.get("MODEL_PATH", "saved_models/best_pneumonia_cnn.keras") # Prioritize env var
# For now, let's hardcode for simplicity in this step, but acknowledge it should be configurable.
MODEL_PATH = "saved_models/best_pneumonia_cnn.keras" # Or pneumonia_cnn_v1.keras if best_ is not there yet
IMAGE_HEIGHT = 150 # Should match training
IMAGE_WIDTH = 150  # Should match training
TARGET_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] # Ensure this order matches model training output

app = Flask(__name__)

# Load the trained model
# Wrap in a try-except block for robustness
try:
    if not os.path.exists(MODEL_PATH):
        # This check is more for immediate feedback; the main check is in __main__ for app startup
        print(f"Warning: Model file {MODEL_PATH} not found during initial script load. Will attempt load in __main__.")
        model = None
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model during initial script load: {e}")
    model = None # Set model to None if loading fails

def preprocess_image(image_bytes, target_size):
    '''
    Preprocesses the input image bytes to the format expected by the model.
    - Decodes image bytes.
    - Converts to RGB if it has an alpha channel.
    - Resizes to target_size.
    - Converts to numpy array and normalizes pixel values (0-1).
    - Adds batch dimension.
    '''
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Handle RGBA to RGB conversion (e.g. for PNGs)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    global model # Allow modification of the global 'model' variable if re-attempting load
    if model is None:
        # Attempt to reload model if it failed during initial load, e.g., file wasn't ready.
        # This is a simple retry, more robust mechanisms might be needed in a production app.
        print("Model was None, attempting to reload...")
        try:
            if not os.path.exists(MODEL_PATH):
                 return jsonify({"error": f"Model file still not found at {MODEL_PATH}."}), 500
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model re-loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error re-loading model: {e}")
            return jsonify({"error": "Model not loaded. Check server logs."}), 500
    
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            image_bytes = file.read()
            processed_image = preprocess_image(image_bytes, TARGET_SIZE)

            if processed_image is None:
                return jsonify({"error": "Image preprocessing failed."}), 400

            prediction = model.predict(processed_image)
            
            # Assuming binary classification with sigmoid output
            # prediction[0][0] gives the probability of the positive class (e.g., PNEUMONIA)
            probability_pneumonia = float(prediction[0][0]) 
            
            if probability_pneumonia > 0.5:
                predicted_class_index = 1 # PNEUMONIA
            else:
                predicted_class_index = 0 # NORMAL
            
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            
            # More detailed probability for the predicted class
            # If PNEUMONIA (prob > 0.5), prob_for_class = prob_pneumonia
            # If NORMAL (prob <= 0.5), prob_for_class = 1 - prob_pneumonia
            prob_for_class = probability_pneumonia if predicted_class_index == 1 else 1 - probability_pneumonia

            return jsonify({
                "class": predicted_class_name,
                "probability": round(prob_for_class, 4) 
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": "Prediction failed. Check server logs."}), 500
    
    return jsonify({"error": "Unknown error"}), 500

if __name__ == "__main__":
    # Create saved_models directory if it doesn't exist, as the model path points there.
    os.makedirs("saved_models", exist_ok=True) 
    
    print(f"Attempting to load model from: {os.path.abspath(MODEL_PATH)}")
    if model is None: # If initial load failed
        try:
            if not os.path.exists(MODEL_PATH):
                print(f"Warning: Model file {MODEL_PATH} does not exist. The API will not work until the model is available.")
            else:
                model = tf.keras.models.load_model(MODEL_PATH)
                print(f"Model loaded successfully from {MODEL_PATH} in __main__ block.")
        except Exception as e:
            print(f"Error loading model in __main__ block: {e}. The API will not work until the model is available.")
            # Model remains None, predict route will handle this.

    app.run(debug=True, host='0.0.0.0', port=5000)
