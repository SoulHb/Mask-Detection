import argparse
import cv2
import numpy as np
import io
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from config import *

app = Flask(__name__)


def visualize_detection(image, detections):
    """
        Visualizes the YOLO model detections on the input image.

        Args:
            image (PIL.Image): Input image.
            detections (list): List of bounding boxes and class probabilities.

        Returns:
            np.array: Image with bounding boxes drawn.
        """
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    class_names = [("without_mask",(0, 0, 255)) , ("with_mask", (255, 0, 0)), ("mask_worn_incorrect",(0, 255, 0))]

    for bbox in detections:
        x_min, y_min, x_max, y_max, prob_class, class_id = bbox
        class_id = int(class_id)
        label = f"{class_names[class_id]}: {prob_class:.2f}"
        color = class_names[class_id][1]
        thickness = 1
        font_size = 0.3
        font_thickness = 1
        text_offset = 2

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
        cv2.rectangle(image, (int(x_min), int(y_min - text_size[1] - text_offset)),
                      (int(x_min) + text_size[0], int(y_min)), (0, 0, 255), cv2.FILLED)

        # Draw white text above the bounding box
        cv2.putText(image, label, (int(x_min), int(y_min - text_offset)), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), font_thickness)

        # Draw the bounding box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@app.route('/process_image', methods=['POST'])
def run_inference_img():
    """
        Endpoint for processing a single image and returning the visualized result.

        Returns:
            jsonify: JSON response with the visualized prediction.
        """
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']
    image = Image.open(file.stream)
    results = model(image)

    image = visualize_detection(image, results[0].boxes.data.tolist())

    return jsonify({'prediction': image.tolist()})


@app.route('/process_video', methods=['POST'])
def run_inference_video():
    """
        Endpoint for processing a video file, adding bounding boxes, and returning the result as a video.

        Returns:
            tuple: Video data, HTTP status code, and content type header.
        """
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']

    # Save the video file
    file.save(file.filename)

    # Open the video file
    cap = cv2.VideoCapture(file.filename)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
    out = cv2.VideoWriter('output.webm', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run inference on the frame using the model
        results = model(image)

        # Get the bounding boxes from the results
        bboxes = results[0].boxes.data.tolist()

        class_names = ["without_mask", "with_mask", "mask_weared_incorrect"]

        # Draw bounding boxes on the frame
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, prob_class, class_id = bbox
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
            label = f"{class_names[int(class_id)]}: {prob_class:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (int(x_min), int(y_min) - text_size[1]),
                          (int(x_min) + text_size[0], int(y_min)), color, cv2.FILLED)
            cv2.putText(frame, label, (int(x_min), int(y_min)), font, font_scale, (255, 255, 255), font_thickness)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release the video capture and writer
    cap.release()
    out.release()

    # Return the output video file
    with open('output.webm', 'rb') as f:
        video_data = f.read()

    return video_data, 200, {'Content-Type': 'video/mp4'}


def main(args):
    """
        Main function to initialize the YOLO model and run the Flask app.

        Args:
            args (dict): Dictionary containing command-line arguments or default values.
                Possible keys: 'saved_model_path'.
        """
    global model
    model = YOLO(args['saved_model_path'] if args['saved_model_path'] else f'{SAVED_MODEL_PATH}/model.pt')
    model.to(DEVICE)
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str, help='Specify path for saved model')
    args = parser.parse_args()
    args = vars(args)
    main(args)