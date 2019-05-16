import os
import sys
import random
import cv2

import pandas as pd

from pascal_voc_tools import XmlReader
from collections import defaultdict

def prepare_dataset(data_dir):
    if not os.path.exists('image_files'):
        os.makedirs('image_files')
    pascal_dir = os.path.join(data_dir, 'outputs')
    pascal_list = [file for file in os.listdir(pascal_dir) if os.path.isfile(os.path.join(pascal_dir, file))]
    metadata = defaultdict(list)
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.PNG'
    for pascal in pascal_list:
        reader = XmlReader(os.path.join(pascal_dir, pascal))
        ann_dict = reader.load()
        for ann in ann_dict['object']:
            new_file_name = fn(ann_dict['filename'])
            metadata['image_name'].append(new_file_name)
            metadata['label'].append(ann['name'])
            crop_save_img(os.path.join(data_dir, ann_dict['filename']), new_file_name, ann['bndbox'])
    df = pd.DataFrame(metadata).to_csv('metadata.csv')

def crop_save_img(file_path, new_name, bndbox):
    img = cv2.imread(file_path)
    img2 = img[int(bndbox['ymin']):int(bndbox['ymax']), int(bndbox['xmin']):int(bndbox['xmax'])]
    cv2.imwrite(os.path.join('image_files', new_name), img2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare dataset")

    parser.add_argument('--dir', '-d',
        help="directory with images"
    )
    
    args = parser.parse_args()
    prepare_dataset(args.dir)