import os
import argparse
from typing import Callable, Dict, List, Union

from tqdm import tqdm

from common import FileList, FuncFactory


K_YOLO = 'yolo'

parserFactory = FuncFactory()
mapFactory = FuncFactory()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to root of labels')
    parser.add_argument('-t', '--type', type=str, required=True, help='type of labels, such as yolo')
    ### more arguments ###
    args = parser.parse_args()
    return args


@parserFactory.register(K_YOLO)
def parse_yolo_label(path_label: str) -> Dict[str, int]:
    '''
    解析一个yolo类型的标签文件的类别数量
    '''
    cls_cnt = dict()
    with open(path_label, encoding='utf-8') as f:
        content = f.read().strip()
    lines = content.splitlines()
    for line in lines:
        _cls = line.split()[0]
        if _cls not in cls_cnt:
            cls_cnt[_cls] = 1
        else:
            cls_cnt[_cls] += 1
    return cls_cnt


@mapFactory.register(K_YOLO)
def map_yolo_label(path_input: str) -> Dict[str, str]:
    '''
    标签映射, 用于替换原类别计数器里的类别名称
    @path_input: 根目录
    @return: dict{原类别 : 替换的类别}
    '''
    classes_ind_path = os.path.join(path_input, 'classes.txt')
    label_map = dict()
    with open(classes_ind_path, encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        label_map[str(i)] = line
    return label_map


def statistic_labels(labels: List[str], cls_cnt: Callable) -> Dict[str, int]:
    '''
    解析标签, 统计标签中各个类别的标注框数目
    @labels: 标签路径的列表
    @cls_cnt: 计算每个标签类别个数的函数
    @return: 保存每个类别对应框的个数的字典 
    '''
    all_cls_cnt = dict()
    for label in tqdm(labels):
        lab_cls_cnt = cls_cnt(label)
        for _cls in lab_cls_cnt:
            if _cls in all_cls_cnt:
                all_cls_cnt[_cls] += lab_cls_cnt[_cls]
            else:
                all_cls_cnt[_cls] = lab_cls_cnt[_cls]
    return all_cls_cnt


def main(args):
    fileList = FileList(args.input)
    labels = fileList.exclude(fileList.file('classes.txt'), fileList.file('.DS_Store'))
    cls_cnt = parserFactory.getFunction(args.type)
    all_cls_cnt = statistic_labels(labels, cls_cnt)
    lab_map = mapFactory.getFunction(args.type)(args.input)
    for cls in all_cls_cnt:
        if lab_map:
            print(f'{lab_map[cls]} : {all_cls_cnt[cls]}')
        else:
            print(f'{cls} : {all_cls_cnt[cls]}')


if __name__ == '__main__':
    args = get_args()
    print(f'Input: {args.input}')
    print(f'Type: {args.type}')
    main(args)

