import json
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from shutil import copy

import yaml


# label: 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip'
# train: 'http://images.cocodataset.org/zips/train2017.zip'  # 19G, 118k images
# val:   'http://images.cocodataset.org/zips/val2017.zip'    # 1G, 5k images


def yaml_save(file='data.yaml', data=None):
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        # Dump data to file in YAML format, converting Path objects to strings
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False, allow_unicode=True)


def get_specify_target_on_coco(save_path, coco_path, specify):
    """Extracts the specified categories from the COCO dataset and generates a data profile"""

    assert os.path.isdir(save_path), 'save path is not a directory'
    assert os.path.isdir(coco_path), 'coco path is not a directory'

    # get coco categories
    coco_cls = {}
    with open(os.path.join(coco_path, 'annotations/instances_val2017.json'), 'r') as fp:
        coco_categories_dict = json.load(fp)['categories']
    for obj in coco_categories_dict:
        coco_cls[obj['name']] = obj['id'] - 1

    try:
        specify_id = [coco_cls[cls] for cls in specify]
    except KeyError as e:
        print(f'specify value ({e}) must is str, and in coco categories')
        exit()

    # get specify labels on coco
    coco_label_paths = {'train': os.path.join(coco_path, 'labels/train2017'), 'val': os.path.join(coco_path, 'labels/val2017')}
    save_label_paths = {'train': os.path.join(save_path, 'labels/train'), 'val': os.path.join(save_path, 'labels/val')}
    os.makedirs(save_label_paths['train'], exist_ok=True)
    os.makedirs(save_label_paths['val'], exist_ok=True)

    def _get_specify_target(save_label_path, coco_label_files):
        for lab_fp in coco_label_files:
            with open(lab_fp, 'r') as fp:
                objs = fp.readlines()
            specify_objs = []
            for obj in objs:
                cls_id = int(obj.split(' ')[0])
                if cls_id in specify_id:
                    specify_objs.append('{}{}'.format(specify_id.index(cls_id), obj[1:]))
            if specify_objs:
                with open(os.path.join(save_label_path, os.path.basename(lab_fp)), 'w+') as fp:
                    for obj in specify_objs:
                        fp.write(obj)

    work_num = os.cpu_count()
    with ThreadPool(work_num) as pool:
        coco_label_files = [os.path.join(coco_label_paths['train'], lab_fp) for lab_fp in os.listdir(coco_label_paths['train'])]
        per_take_number = len(coco_label_paths['train']) // work_num + 1
        if per_take_number > 100:
            for i in range(work_num):
                pool.apply_async(_get_specify_target, args=(save_label_paths['train'], coco_label_files[i * per_take_number: (i + 1) * per_take_number]))
            pool.close()
            pool.join()
        else:
            _get_specify_target(save_label_paths['train'], coco_label_files)
    coco_label_files = [os.path.join(coco_label_paths['val'], lab_fp) for lab_fp in os.listdir(coco_label_paths['val'])]
    _get_specify_target(save_label_paths['val'], coco_label_files)

    # copy specify image from coco
    coco_image_paths = {'train': os.path.join(coco_path, 'images/train2017'), 'val': os.path.join(coco_path, 'images/val2017')}
    save_image_paths = {'train': os.path.join(save_path, 'images/train'), 'val': os.path.join(save_path, 'images/val')}
    os.makedirs(save_image_paths['train'], exist_ok=True)
    os.makedirs(save_image_paths['val'], exist_ok=True)

    def _copy_images_on_coco(save_image_path, coco_image_paths):
        for coco_im_fp in coco_image_paths:
            try:
                copy(coco_im_fp, save_image_path)
            except IOError as e:
                print(e)

    with ThreadPool(work_num) as pool:
        coco_image_files = [os.path.join(coco_image_paths['train'], im_fp.replace('txt', 'jpg')) for im_fp in os.listdir(save_label_paths['train'])]
        per_take_number = len(coco_image_files) // work_num + 1
        if per_take_number > 100:
            for i in range(work_num):
                pool.apply_async(_copy_images_on_coco, args=(save_image_paths['train'], coco_image_files[i * per_take_number: (i + 1) * per_take_number]))
            pool.close()
            pool.join()
        else:
            _copy_images_on_coco(save_image_paths['train'], coco_image_files)

    coco_image_files = [os.path.join(coco_image_paths['val'], im_fp.replace('txt', 'jpg')) for im_fp in os.listdir(save_label_paths['val'])]
    _copy_images_on_coco(save_image_paths['val'], coco_image_files)

    # create train.txt and val.txy file
    with open(os.path.join(save_path, 'train.txt'), 'w+') as fp:
        for im_name in os.listdir(save_image_paths['train']):
            fp.write(f'./images/train/{im_name}\n')

    with open(os.path.join(save_path, 'val.txt'), 'w+') as fp:
        for im_name in os.listdir(save_image_paths['val']):
            fp.write(f'./images/val/{im_name}\n')

    # # create data_yaml file
    yaml_data = {
        'path':  save_path,
        'train': 'train.txt',
        'val':   'val.txt',
        'test':  'none',
        'names': {}
    }
    for i, cls in enumerate(specify):
        yaml_data['names'][i] = cls

    yaml_save(os.path.join(Path(save_path).parent, '{}.yaml'.format(Path(save_path).name)), yaml_data)


if __name__ == '__main__':
    save_path = '../data/vehicle'
    os.makedirs(save_path, exist_ok=True)
    coco_path = '../data/coco'
    specify = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']

    get_specify_target_on_coco(save_path, coco_path, specify)

    print('done')
