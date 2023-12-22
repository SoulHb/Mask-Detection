import yaml
import torch
import gdown
import os
import sys
sys.path.append('src')


def create_save_result_folder(path):
    """
        Create a folder at the specified path if it does not exist.

        Args:
            path (str): The path to the folder.
        Return: None
        """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")


def load_model(url, output_path):
    """
        Download a model from the given URL and save it to the specified output path
        if the model does not exist at the output path.

        Args:
            url (str): The URL from which to download the model.
            output_path (str): The path to save the downloaded model.
        Return: None
        """
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
        print("Model loaded!")
    else:
        print(f"Model '{output_path}' already exists.")


with open('./config.yaml', 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file)

# Access the variables in upper case
EPOCHS = config_data.get('epochs', None)
BATCH_SIZE = config_data.get('batch_size', None)
MODEL_NAME = config_data.get('model_name', None)
IMAGE_SIZE = (config_data.get('image_height', None), config_data.get('image_width', None))
IMAGES_PATH = config_data.get('images_path', None)
ANNOTATIONS_PATH = config_data.get('annotations_path', None)
DATA_PATH = config_data.get('data_path', None)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_URL = config_data.get('model_url', None)
SAVED_MODEL_PATH = config_data.get('saved_model_path', None)
create_save_result_folder(SAVED_MODEL_PATH)
load_model(url=MODEL_URL, output_path=os.path.join(SAVED_MODEL_PATH, 'model.pt'))