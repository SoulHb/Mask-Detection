from ultralytics import YOLO
import albumentations as A
import torch
from dataset import create_data
image_height, image_width = 640, 640


create_data()
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=image_height, batch=2)