import os
import shutil

from collections import defaultdict
from hashlib import md5
from pathlib import Path

import PIL

# Remove file function
def remove_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)

# SET THE DIRECTORY PATH   
raw_img_dir = './dataset/images'
deduped_img_dir = './dataset/deduped_images'

# CHECKING OF DUPLICATES
# IF FOUND, SAVE TO duplicate_all and for_removal as a list
image_dir = Path(raw_img_dir)

hash_dict = defaultdict(list)
for image in image_dir.glob('*.jpg'):
    with image.open('rb') as f:
        img_hash = md5(f.read()).hexdigest()
        hash_dict[img_hash].append(image)

duplicate_all = []
for_removal = []
for k, v in hash_dict.items():
    duplicate_pair = []

    if len(v) > 1:
        if v[0].name != v[1].name:
            duplicate_pair.append(v[0])
            duplicate_pair.append(v[1])

            for_removal.append(v[1])
    
        duplicate_all.append(duplicate_pair)

# Change the data type from Posixpath to String
for_removal = list(map(str, for_removal))

# Save to txt file
with open('duplicates.txt', 'w') as f:
    for image in for_removal:
        name = os.path.basename(image)
        f.write(f"{name}\n")


# Copy all the files from the downloaded images to a new directory
if not os.path.exists(deduped_img_dir):
    shutil.copytree(raw_img_dir, deduped_img_dir)
else:
    shutil.rmtree(deduped_img_dir)
    shutil.copytree(raw_img_dir, deduped_img_dir)

# Remove all the items in for_removal list from the new directory
for dup in for_removal:
    name = os.path.basename(dup)
    remove_file(name, deduped_img_dir)

    print(f'File {name} is removed from {deduped_img_dir}')




