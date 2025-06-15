Football Offside Detection System.
This project implements a computer vision pipeline for detecting football players, extracting their keypoints, computing vanishing points, and drawing virtual offside lines based on player pose and camera perspective.

📸 Demo Output
Detects pitch and players

Computes vanishing point from pitch lines

Classifies players by team (based on jersey color)

Extracts keypoints (shoulders, knees, ankles) using Keypoint R-CNN

Draws virtual offside lines toward the vanishing point

🧰 Features
✅ Pitch segmentation using Sobel edges and clustering

✅ Player detection with YOLOv8

✅ Pose estimation using keypointrcnn_resnet50_fpn

✅ Team classification using KMeans or GMM

✅ Vanishing point estimation from pitch lines

✅ Offside line visualization from selected keypoints to the vanishing point

🛠️ Requirements
Install dependencies via:
```
pip install -r requirements.txt
```

🧾 requirements.txt
```
  arduino
  Copy
  Edit
  pandas
  opencv-python
  numpy
  torch
  torchvision
  matplotlib
  ultralytics
  scikit-learn
  pclines
  scikit-image
  shapely
```

🚀 How to Run
Place your match image under datasets/images/ e.g., liv.jpg.

Run the main pipeline:
```
image_path = "datasets/images/liv.jpg"
original, pitch_only, pitch_mask = segment_pitch_area(image_path)
edges = extract_edges_sobel(pitch_only, pitch_mask)
final_img, final_lines = detect_lines_with_clustering(edges, pitch_only)

# Compute vanishing point
vanishing_point = compute_vanishing_point(final_lines)
plot_lines_and_vp(original, final_lines, vanishing_point)

# Detect players and extract keypoints
img_with_players, player_boxes = detect_players_yolov8(pitch_only)
classified_img, team_labels, filtered_boxes = classify_players_kmeans(img_with_players, player_boxes)
keypoints, pose_output = extract_keypoints_for_detected_players(img_with_players, filtered_boxes)

# Draw offside lines
draw_offside_lines(pose_output, keypoints, vanishing_point)
```
