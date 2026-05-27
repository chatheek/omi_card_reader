from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_FILE = "omi_svm_model.xml"
labels_map = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']

# Initialize model into cloud memory layer on container startup
if os.path.exists(MODEL_FILE):
    svm = cv2.ml.SVM_load(MODEL_FILE)
    hog = cv2.HOGDescriptor(_winSize=(20,30), _blockSize=(10,10), _blockStride=(5,5), _cellSize=(5,5), _nbins=9)
    print("🚀 Cloud Engine Initialized Safely.")
else:
    print("❌ Critical Error: omi_svm_model.xml is missing from the workspace root folder!")
    svm = None

@app.route('/predict', methods=['POST'])
def predict_card():
    if svm is None:
        return jsonify({"error": "AI Engine missing"}), 500
        
    try:
        # Check if client sent raw payload bytes or standard multi-part data form
        if request.data:
            file_bytes = np.frombuffer(request.data, dtype=np.uint8)
        elif 'image' in request.files:
            file_bytes = np.frombuffer(request.files['image'].read(), dtype=np.uint8)
        else:
            return jsonify({"status": "NO_IMAGE"}), 400
            
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "DECODE_FAIL"}), 400
            
        frame_resized = cv2.resize(frame, (320, 240))
        h, w, _ = frame_resized.shape
        
        # Crop out your 40x60 'Hot Zone' straight from frame center
        box_w, box_h = 40, 60
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        zone_roi = frame_resized[y1:y1+box_h, x1:x1+box_w]
        
        # Universal red/black filter channel isolation
        b, g, r = cv2.split(zone_roi)
        dark_pass = cv2.min(cv2.min(b, g), r)
        
        blurred = cv2.medianBlur(dark_pass, 3)
        thresh_zone = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 4)
        
        # Apply the strong horizontal bridge logic you built to stitch '1' and '0'
        strong_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        dilated_zone = cv2.dilate(thresh_zone, strong_bridge_kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        max_area = 0
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if cw > 5 and ch > 15:
                area = cw * ch
                if area > max_area:
                    max_area = area
                    best_contour = (cx, cy, cw, ch)
                    
        if best_contour is not None:
            cx, cy, cw, ch = best_contour
            char_roi = thresh_zone[cy:cy+ch, cx:cx+cw]
            resized_roi = cv2.resize(char_roi, (20, 30))
            
            descriptor = hog.compute(resized_roi).reshape(1, -1)
            result = svm.predict(descriptor)[1]
            class_id = int(result[0][0])
            
            if 0 <= class_id < len(labels_map):
                return jsonify({"card": labels_map[class_id]}), 200
                
        return jsonify({"card": "NONE"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)