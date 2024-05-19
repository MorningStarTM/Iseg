import os 
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "Iseg"

list_of_files = [

    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/unet/__init__.py",
    f"src/{project_name}/models/segnet/__init__.py",
    f"src/{project_name}/models/attention/__init__.py",
    f"src/{project_name}/models/transformer/__init__.py",
    f"src/{project_name}/models/state_space_models/__init__.py",
    f"src/{project_name}/models/deep/__init__.py",
    f"src/{project_name}/models/application/__init__.py",
    f"src/{project_name}/models/research/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/examples/example_1.ipynb",
    f"src/{project_name}/examples/example_2.ipynb",
    f"src/{project_name}/logging/__init__.py",
    "main.py",
    "requirements.py",
    "setup.py",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory : {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")

        