import math
from pathlib import Path
import argparse
import random
from typing import List

from common import FileList, print_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, required=True, help='path to images')
    parser.add_argument('-l', '--labels', type=str, required=True, help='path to labels')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to output')
    parser.add_argument('-s', '--split', type=str, default='8:1:1', help='train:val:test')
    ### more arguments ###
    args = parser.parse_args()
    return args


def list_images_with_label(images: List[str], labels: List[str]) -> List[str]:
    '''
    列出有标签的图像
    @images: 图像路径列表
    @labels: 标签路径列表
    @return: 带有标签的图像路径列表
    '''
    if len(images) == 0 or len(labels) == 0:
        return []
    images_with_label = []
    labels = [Path(label).stem for label in labels]
    for image in images:
        if Path(image).stem in labels:
            images_with_label.append(image)
    return images_with_label


def split_data(data: List[str], split: str) -> List[List[str]]:
    '''
    将数据按照一定比例进行分配
    @data: 数据路径列表
    @split: data1:data2:data3...
    @return: list, data1:data2:data3...
    '''
    split_list = []
    random.shuffle(data)
    ns = [float(d) for d in split.split(':')] # number of each dataset
    end = 0
    for i in range(len(ns)):
        p = ns[i] / sum(ns)
        start = end
        end = start + math.ceil(p * len(data))
        split_list.append(data[start:end])
    return split_list


def create_data_index(data: List[str], path: str):
    '''
    创建数据路径索引
    @data: 数据路径列表
    @path: 保存索引路径
    '''
    p = Path(path)
    assert not p.exists(), f'{path} is already exists'
    p.touch()
    content = '\n'.join(data)
    p.write_text(content)


def main(args):
    imgList = FileList(args.images)
    labList = FileList(args.labels)
    images_with_label = list_images_with_label(imgList.images(), labList.suffixes('.txt'))
    print(f'find {len(images_with_label)} images with label')
    assert len(args.split.split(':')) == 3, f'{args.split} is illegal'
    splitted_data = split_data(images_with_label, args.split)
    splitted_data_name = ['train.txt', 'val.txt', 'test.txt']
    for i, data in enumerate(splitted_data):
        create_data_index(data, str(Path(args.output) / splitted_data_name[i]))


if __name__ == '__main__':
    args = get_args()
    print_args(args)
    main(args)
