import os
import subprocess
import sys
import zipfile

workspace_dir = '/root/video-generation-model'

if os.getcwd() != workspace_dir:
    os.chdir(workspace_dir)

print(f"Current working directory: {os.getcwd()}")

target_data_dir_root = 'hmdb51_root'

print(f"Creating target directory: {target_data_dir_root}...")
os.makedirs(target_data_dir_root, exist_ok=True)

print("Downloading hmdb51.zip to root...")
download_url = "https://huggingface.co/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip?download=true"
zip_file = "hmdb51.zip"

result = subprocess.run(['wget', download_url, '-O', zip_file], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error downloading: {result.stderr}")
    sys.exit(1)

print(f"Extracting hmdb51.zip to {target_data_dir_root}...")
try:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_data_dir_root)
    print("Extraction completed successfully")
except Exception as e:
    print(f"Error extracting: {e}")
    sys.exit(1)

print("Removing hmdb51.zip...")
os.remove(zip_file)

print(f"Verifying presence of .avi files in {target_data_dir_root}...")
avi_files = []
for root, dirs, files in os.walk(target_data_dir_root):
    for file in files:
        if file.endswith('.avi'):
            avi_files.append(os.path.join(root, file))
            if len(avi_files) >= 5:
                break
    if len(avi_files) >= 5:
        break

for avi_file in avi_files[:5]:
    print(f"  {avi_file}")

print("Manual download and extraction to root directory complete.")

