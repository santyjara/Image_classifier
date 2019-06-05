import os
import sys
import random
import cv2
import json 

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pascal_voc_tools import XmlReader
from collections import defaultdict

labels = {
    "a" : 0,
    "e" : 1,
    "i" : 2,
    "o" : 3,
    "u" : 4
}

def prepare_dataset(data_dir):
    if not os.path.exists('image_files'):
        os.makedirs('image_files')
    #with open('labels.json') as f:
    #    labels = json.load(f)
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
            metadata['label'].append(labels[ann['name']])
            crop_save_img(os.path.join(data_dir, ann_dict['filename']), new_file_name, ann['bndbox'])
    metadata['split'] = split_dataset(len(metadata['label']))
    pd.DataFrame(metadata).to_csv('metadata.csv', index=False)
    metadata = pd.DataFrame(metadata)
    metadata1 = []
    for _, value in labels.items():
        metadata1.append(metadata.query("split == 'train' & label == " + str(value)).iloc[0]) 
    pd.DataFrame(metadata1).to_csv('metadata1.csv', index=False)

def split_dataset(ds_len):
    test =  int(ds_len * 0.2)
    split = ['test'] * test
    train = int(ds_len * 0.7)
    split.extend(['train'] * train)
    val = ds_len - (test + train)
    split.extend(['val'] * val)
    random.shuffle(split)
    return split

def crop_save_img(file_path, new_name, bndbox):
    img = cv2.imread(file_path)
    img2 = img[int(bndbox['ymin']):int(bndbox['ymax']), int(bndbox['xmin']):int(bndbox['xmax'])]
    cv2.imwrite(os.path.join('image_files', new_name), img2)

def build_sources_from_metadata(metadata, data_dir, mode='train', exclude_labels=None):
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df = df[df['split'] == mode]
    df['filepath'] = df['image_name'].apply(lambda x: os.path.join(data_dir, x))
    include_mask = df['label'].apply(lambda x: x not in exclude_labels)
    df = df[include_mask]

    sources = list(zip(df['filepath'], df['label']))
    return sources

def preprocess_image(image):
    image = tf.image.resize(image, size=(32, 32))
    image = image / 255.0
    return image

def make_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (filepath, label_id) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.

    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load(row):
        filepath = row['image']
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        return img, row['label']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, labels = zip(*sources)
    
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'label': list(labels)}) 

    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(lambda x,y: (preprocess_image(x), y))
    
    if training:
        ds = ds.map(lambda x,y: (augment_image(x), y))
        
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

def imshow_batch_of_three(batch):
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        axarr[i].set(xlabel='label = {}'.format(label_batch[i]))

def augment_image(image):
    return image

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare dataset")

    parser.add_argument('--dir', '-d',
        help="directory with images"
    )
    
    args = parser.parse_args()
    prepare_dataset(args.dir)