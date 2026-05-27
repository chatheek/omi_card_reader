from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_FILE = "omi_svm_model.xml"
labels_map = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']

# Global thread-safe variable to hold the latest frame image for visual browser inspection
latest_jpeg_bytes = None

if os.path.exists(MODEL_FILE):
    svm = cv2.ml.SVM_load(MODEL_FILE)
    hog = cv2.HOGDescriptor(_winSize=(20,30), _blockSize=(10,10), _blockStride=(5,5), _cellSize=(5,5), _nbins=9)
    print("🚀 Cloud Inference Engine Successfully Configured with Live Preview Mode.")
else:
    print("❌ Critical Deployment Error: omi_svm_model.xml not found!")
    svm = None

@app.route('/predict', methods=['POST'])
def predict_card():
    global latest_jpeg_bytes
    if svm is None:
        return jsonify({"error": "AI Engine Uninitialized"}), 500
        
    try:
        file_bytes = np.frombuffer(request.data, dtype=np.uint8)
        if len(file_bytes) == 0:
            return jsonify({"card": "NONE", "error": "Empty data payload"}), 400
            
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"card": "NONE", "error": "JPEG decoding failure"}), 400
            
        frame_resized = cv2.resize(frame, (320, 240))
        h, w, _ = frame_resized.shape
        
        box_w, box_h = 40, 60
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        
        # --- DRAW VISUAL GUIDES FOR PREVIEW CHANNEL ---
        # Before cropping the ROI, draw a blue bounding box and target text 
        # onto a copy of the frame to help you align things during testing.
        preview_frame = frame_resized.copy()
        cv2.rectangle(preview_frame, (x1, y1), (x1 + box_w, y1 + box_h), (255, 0, 0), 2)
        cv2.putText(preview_frame, "HOT ZONE", (x1 - 15, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Save this visual frame to global memory
        _, encoded_img = cv2.imencode('.jpg', preview_frame)
        latest_jpeg_bytes = encoded_img.tobytes()
        
        # --- CONTINUE WITH CLEAN PROCESSING ANALYSIS ---
        zone_roi = frame_resized[y1:y1+box_h, x1:x1+box_w]
        
        b, g, r = cv2.split(zone_roi)
        dark_pass = cv2.min(cv2.min(b, g), r)
        blurred = cv2.medianBlur(dark_pass, 3)
        thresh_zone = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 4)
        
        strong_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        dilated_zone = cv2.dilate(thresh_zone, strong_bridge_kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_predicted_label = "NONE"
        
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
                card_predicted_label = labels_map[class_id]
                
        return jsonify({"card": card_predicted_label}), 200
        
    except Exception as e:
        return jsonify({"card": "NONE", "error": str(e)}), 500

# --- NEW ROUTE: STREAM THE PREVIEW LIVE TO YOUR BROWSER ---
@app.route('/preview')
def live_preview():
    if latest_jpeg_bytes is None:
        return "Waiting for first camera frame transmission snapshot...", 200
        
    # Serve a simple auto-refreshing page layout to look at the stream alignment
    html_page = """
    <html>
        <head>
            <title>Omi Cam Testing Portal</title>
            <meta http-equiv="refresh" content="1"> <style>
                body { font-family: Arial, sans-serif; text-align: center; background: #222; color: #fff; padding-top: 50px; }
                img { border: 4px solid #444; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
                h1 { color: #00ffcc; }
            </style>
        </head>
        <body>
            <h1>📷 Live Cloud Testing Feed</h1>
            <p>Position your card rank element straight inside the blue box.</p>
            <img src="/preview/frame.jpg?cache_burst="""" + str(time.time()) + """" />
        </body>
    </html>
    """
    return html_page

@app.route('/preview/frame.jpg')
def preview_frame_bytes():
    if latest_jpeg_bytes is None:
        return "No image data", 404
    return Response(latest_jpeg_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    import time
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)