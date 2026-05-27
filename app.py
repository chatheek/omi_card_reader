from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_FILE = "omi_svm_model.xml"
labels_map = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']

# --- INITIALIZE MACHINE LEARNING MODEL ON STARTUP ---
if os.path.exists(MODEL_FILE):
    svm = cv2.ml.SVM_load(MODEL_FILE)
    # Recreate the exact HOG configuration you used to compile card_dataset variations
    hog = cv2.HOGDescriptor(_winSize=(20,30), _blockSize=(10,10), _blockStride=(5,5), _cellSize=(5,5), _nbins=9)
    print("🚀 Cloud Inference Engine Successfully Configured.")
else:
    print("❌ Critical Deployment Error: omi_svm_model.xml not found in root path!")
    svm = None

@app.route('/predict', methods=['POST'])
def predict_card():
    if svm is None:
        return jsonify({"error": "AI Engine Uninitialized"}), 500
        
    try:
        # Decode the incoming raw JPEG binary stream payload sent by the ESP32-CAM
        file_bytes = np.frombuffer(request.data, dtype=np.uint8)
        if len(file_bytes) == 0:
            return jsonify({"card": "NONE", "error": "Empty data payload"}), 400
            
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"card": "NONE", "error": "JPEG decoding failure"}), 400
            
        # Match your exact internal processing canvas dimension environment
        frame_resized = cv2.resize(frame, (320, 240))
        h, w, _ = frame_resized.shape
        
        # Isolate your precise 40x60 'Hot Zone' tracking boundaries from center
        box_w, box_h = 40, 60
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        zone_roi = frame_resized[y1:y1+box_h, x1:x1+box_w]
        
        # --- EXECUTE YOUR EXACT EXPERT THRESHOLD MATRIX PIPELINE ---
        # Neutralize contrast differences between red and black ink cards
        b, g, r = cv2.split(zone_roi)
        dark_pass = cv2.min(cv2.min(b, g), r)
        
        blurred = cv2.medianBlur(dark_pass, 3)
        thresh_zone = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 4)
        
        # --- THE STRONG HORIZONTAL DILATION BRIDGE ---
        # Merges separate character strokes (like the '1' and '0' of a 10) into one shape
        strong_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        dilated_zone = cv2.dilate(thresh_zone, strong_bridge_kernel, iterations=1)
        
        # Process contours inside the bridged hot zone
        contours, _ = cv2.findContours(dilated_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_predicted_label = "NONE"
        
        best_contour = None
        max_area = 0
        
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            # Retain your custom structural noise constraints filters
            if cw > 5 and ch > 15:
                area = cw * ch
                if area > max_area:
                    max_area = area
                    best_contour = (cx, cy, cw, ch)
                    
        if best_contour is not None:
            cx, cy, cw, ch = best_contour
            
            # Crop cleanly from thresh_zone so the AI reads original line weights without bloating
            char_roi = thresh_zone[cy:cy+ch, cx:cx+cw]
            resized_roi = cv2.resize(char_roi, (20, 30))
            
            # Extract HOG gradient features vector arrays map
            descriptor = hog.compute(resized_roi).reshape(1, -1)
            
            # Compute classification using the loaded SVM model
            result = svm.predict(descriptor)[1]
            class_id = int(result[0][0])
            
            if 0 <= class_id < len(labels_map):
                card_predicted_label = labels_map[class_id]
                
        # Return a clean payload back to the hardware infrastructure client
        return jsonify({"card": card_predicted_label}), 200
        
    except Exception as e:
        return jsonify({"card": "NONE", "error": str(e)}), 500

if __name__ == '__main__':
    # Grab the dynamic port assigned by your hosting platform (defaults to 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)