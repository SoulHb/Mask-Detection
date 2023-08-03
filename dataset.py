import os
import xml.etree.ElementTree as ET
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

class_mapping = {
    "without_mask": 0,
    "with_mask": 1,
    "mask_weared_incorrect": 2,
    # Add more classes as needed
}


def extract_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_lines = []
    size = root.findall('size')[0]
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = class_mapping[name]
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)/width
        ymin = int(bbox.find('ymin').text)/height
        xmax = int(bbox.find('xmax').text)/width
        ymax = int(bbox.find('ymax').text)/height
        yolo_lines.append([xmin, ymin, xmax, ymax, label])

    # Get the name of the XML file (without extension) to use as the name for the YOLO format text file.
    file_name_without_extension = os.path.splitext(os.path.basename(xml_file))[0]
    image_file_name = f"{file_name_without_extension}.png"

    return image_file_name, yolo_lines


def save_files(files, src_folder, dest_folder, images_path):
    for file in files:
        xml_file_path = os.path.join(src_folder, file)
        image_file_name, yolo_lines = extract_annotations(xml_file_path)

        # Get the name of the XML file (without extension) to use as the name for the YOLO format text file.
        file_name_without_extension = os.path.splitext(file)[0]
        yolo_file_name = f"{file_name_without_extension}.txt"
        with open(os.path.join(dest_folder, "labels", yolo_file_name), "w") as f:
            for bbox in yolo_lines:
                label = bbox[4]
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                w = (bbox[2] - bbox[0])
                h = (bbox[3] - bbox[1])
                f.write(f"{label} {x_center} {y_center} {w} {h}\n")
        image_path = os.path.join(images_path, image_file_name)
        shutil.copy(image_path, os.path.join(dest_folder, "images", image_file_name))


def create_data(destination_folder=r".", images_path=r".\data\images", folder_annotations_path=r".\data\annotations"):
    train_folder = os.path.join(destination_folder, "data/train")
    val_folder = os.path.join(destination_folder, "data/val")
    test_folder = os.path.join(destination_folder, "data/test")
    if not os.path.exists(train_folder):
        # Create train, validation, and test folders if they don't exist
        for folder in [train_folder, val_folder, test_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                os.makedirs(os.path.join(folder, "images"))
                os.makedirs(os.path.join(folder, "labels"))

        # Get a list of all annotation files
        annotation_files = [file for file in os.listdir(folder_annotations_path) if file.endswith(".xml")]

        # Split the annotation files into train, validation, and test sets
        train_files, test_files = train_test_split(annotation_files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

        # Save files to train, validation, and test folders
        save_files(train_files, folder_annotations_path, train_folder, images_path)
        save_files(val_files, folder_annotations_path, val_folder, images_path)
        save_files(test_files, folder_annotations_path, test_folder, images_path)
    else:
        print("Dataset is ready to use!")