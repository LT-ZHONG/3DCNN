import os
import zipfile
from tensorflow.keras.utils import get_file


# Download url of normal CT scans.
url = 'https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip'
file_name = os.path.join(os.getcwd(), 'CT-0.zip')
get_file(fname=file_name, origin=url)

# Download url of abnormal CT scans.
url = 'https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip'
file_name = os.path.join(os.getcwd(), 'CT-23.zip')
get_file(fname=file_name, origin=url)

# Make a directory to store the data.
os.makedirs('MosMedData')

# Unzip data in the newly created directory.
with zipfile.ZipFile('CT-0.zip', 'r') as z_fp:
    z_fp.extractall('./MosMedData/')

with zipfile.ZipFile('CT-23.zip', 'r') as z_fp:
    z_fp.extractall('./MosMedData/')
