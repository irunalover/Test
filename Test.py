import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

app = Flask(__name__)

# Load the ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet')

# Remove the final classification layers to use the model as a feature extractor
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # ResNet50 expects 224x224 input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess the image for ResNet50
    return img

def get_image_features(img_path):
    img = preprocess_image(img_path)
    features = model.predict(img)  # Extract features from the image
    return features

def cosine_similarity(features1, features2):
    dot_product = np.dot(features1, features2.T)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    return dot_product / (norm1 * norm2)

def match_images_using_ai(img1_path, img2_path, threshold=0.9):
    features1 = get_image_features(img1_path)
    features2 = get_image_features(img2_path)
    
    similarity = cosine_similarity(features1, features2)
    
    # Print the cosine similarity for reference
    print(f"Cosine Similarity (AI): {similarity[0][0]}")
    
    # Decide if images match based on threshold
    if similarity[0][0] >= threshold:
        return True  # Images match
    else:
        return False  # Images do not match

def compute_lab_histogram_similarity(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(min(similarity * 100, 100), 0)

def remove_white_background(img):
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)
    return cv2.bitwise_and(img, img, mask=~mask)

def resize_with_padding(img, target_shape):
    """Resize image to fit within target_shape while maintaining aspect ratio and padding with white."""
    target_h, target_w = target_shape
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

@app.route("/")
def home():
    return "ResNet50-based Flask API is running!"

@app.route("/compare", methods=["POST"])
def compare_images():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "Both image files are required."}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    # Save the images temporarily
    img1_path = f"temp_{img1.filename}"
    img2_path = f"temp_{img2.filename}"

    img1.save(img1_path)
    img2.save(img2_path)

    # Perform the AI-based comparison using ResNet50
    match = match_images_using_ai(img1_path, img2_path)

    # Perform SIFT and LAB similarity computation (original logic)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Resize both images to the same shape
    target_shape = (1000, 1000)
    img1 = resize_with_padding(img1, target_shape)
    img2 = resize_with_padding(img2, target_shape)

    # Remove white backgrounds
    img1_no_bg = remove_white_background(img1)
    img2_no_bg = remove_white_background(img2)

    # Grayscale conversion
    gray1 = cv2.cvtColor(img1_no_bg, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_no_bg, cv2.COLOR_BGR2GRAY)

    # SIFT + FLANN matcher
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return {"error": "Could not compute descriptors."}

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    good_matches = [m for m in matches if m.distance < 0.85 * max(m.distance for m in matches)]
    matches = sorted(good_matches, key=lambda x: x.distance)

    total_keypoints = min(len(kp1), len(kp2))
    num_matches = len(matches)

    # Draw match image
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_image_path = os.path.join(os.getcwd(), "static", f"match_{img1.filename}_{img2.filename}.jpg")
    cv2.imwrite(match_image_path, match_img)

    # Final match ratio with 80% boost if good matches >= 500
    base_ratio = (num_matches / total_keypoints) * 100 if total_keypoints > 0 else 0
    final_match_ratio = base_ratio + 80 if num_matches >= 400 else base_ratio
    final_match_ratio = min(final_match_ratio, 100)

    # LAB color similarity using histogram method
    lab_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    lab_similarity = compute_lab_histogram_similarity(lab_img1, lab_img2)

    # Combine the final similarity score (SIFT and LAB)
    final_similarity_score = (final_match_ratio * 0.6) + (lab_similarity * 0.4)

    # Printed Output (Retaining the original format)
    print(f"SIFT Similarity: {final_match_ratio}%")
    print(f"LAB Similarity: {lab_similarity}%")
    print(f"Final Similarity Score: {final_similarity_score}%")
    print(f"The images {'match' if match else 'do not match'}.")

    # Clean up by removing the temporary files
    os.remove(img1_path)
    os.remove(img2_path)

    # Return the final result (Retaining original format)
    return jsonify({
        "sift_similarity": round(final_match_ratio, 2),
        "lab_similarity": round(lab_similarity, 2),
        "final_similarity_score": round(final_similarity_score, 2),
        "good_matches": num_matches,
        "keypoints_image1": len(kp1),
        "keypoints_image2": len(kp2),
        "match_result": "Match" if match else "No Match",
        "match_image_url": f"/static/match_{img1.filename}_{img2.filename}.jpg"
    })

if __name__== "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True host="0.0.0.0", port=5050)