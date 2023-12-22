import sys
import argparse
from ultralytics import YOLO
from dataset import create_data
from config import *
sys.path.append('src')


def main(args):
    """
        Main function for training the YOLO model.

        Args:
            args (dict): Dictionary containing command-line arguments or default values.
                Possible keys: 'epochs', 'batch_size', 'image_height', 'image_width',
                               'data_path', 'model_name'.

        Returns:
            None
        """
    # Extracting parameters from command-line arguments or using default values from config
    epochs = args['epochs'] if args['epochs'] else EPOCHS
    batch_size = args['batch_size'] if args['batch_size'] else BATCH_SIZE
    image_height = args["image_height"] if args["image_height"] else IMAGE_SIZE[0]
    image_width = args["image_width"] if args["image_width"] else IMAGE_SIZE[1]
    data_path = args['data_path'] if args['data_path'] else DATA_PATH
    model_name = args['model_name'] if args['model_name'] else MODEL_NAME

    # Creating dataset (images and annotations)
    create_data(images_path=f"{data_path}/images", folder_annotations_path=f"{data_path}/annotations")

    # Initializing YOLO model
    model = YOLO(model_name)

    # Training the model
    model.train(data='./dataset.yaml', epochs=epochs, imgsz=(image_height, image_width), batch=batch_size)


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", type=str, help='Specify height')
    parser.add_argument("--image_width", type=str, help='Specify width')
    parser.add_argument("--data_path", type=str, help='Specify path for data folder')
    parser.add_argument("--epochs", type=int, help='Specify epoch for model training')
    parser.add_argument("--lr", type=float, help='Specify learning rate')
    parser.add_argument("--batch_size", type=float, help='Specify batch size for training')
    parser.add_argument("--model_name", type=str, help='Specify model that you want to load')

    # Parsing command-line arguments
    args = parser.parse_args()
    args = vars(args)

    # Calling the main function with parsed arguments
    main(args)