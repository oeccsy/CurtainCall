import cv2
import numpy as np

# Load YOLOv3 weights and configuration files
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Set the model to use GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the classes from the COCO dataset
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the image
img = cv2.imread('image.jpg')

# Get the image dimensions
height, width, _ = img.shape

# Create a blob from the input image and pass it to the network
blob = cv2.dnn.blobFromImage(
    img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Run forward pass and get the network output
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Process each output layer
for output in layer_outputs:
    # Process each detection
    for detection in output:
        # Get the class ID and confidence
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the center and dimensions of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw the bounding box and label on the image
            color = colors[class_id]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the output image
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
