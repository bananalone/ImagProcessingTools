import os, argparse
from typing import List

from common import FileList, copy_files, print_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to input, required')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to output, required')
    parser.add_argument('-m', '--merge', action='store_true', help='merge or not')
    parser.add_argument('-n', '--number', type=int, default=1, help='number of person to annotate, default 1')
    ### more arguments ###
    args = parser.parse_args()
    return args


def assign(files: List[str], number: int) -> List[List[str]]:
    '''
    分配标注任务, 将待标注图像根据人数分为不相交的子集
    '''
    assert number <= len(files), f'{number}:number > {len(files)}:len(files)'
    batch_size = len(files) // number if len(files) % number == 0 else len(files) // number + 1
    files = sorted(files)
    filesList = []
    for i in range(number):
        start = i * batch_size
        end = start + batch_size
        filesList.append(files[start : end])
    return filesList


def init_tasks(root: str, output: str, number: int):
    '''
    初始化标注任务
    '''
    assert not os.path.exists(args.output), f'{args.output} already exists'
    os.makedirs(args.output)
    path_task_list = []
    for i in range(number):
        path_task = os.path.join(output, 'T'+str(i))
        os.mkdir(path_task)
        path_task_list.append(path_task)
    fileList = FileList(root)
    imgs = fileList.images()
    imgs_list = assign(imgs, number)
    for i, imgs in enumerate(imgs_list):
        os.mkdir(os.path.join(path_task_list[i], 'images'))
        os.mkdir(os.path.join(path_task_list[i], 'labels'))
        copy_files(imgs, os.path.join(path_task_list[i], 'images'))


def merge_tasks(root: str, output: str):
    '''
    合并标注任务
    '''
    assert not os.path.exists(args.output), f'{args.output} already exists'
    os.makedirs(args.output)
    path_images = os.path.join(output, 'images')
    if not os.path.exists(path_images):
        os.mkdir(path_images)
    path_labels = os.path.join(output, 'labels')
    if not os.path.exists(path_labels):
        os.mkdir(path_labels)
    fileList = FileList(root)
    path_task_list = fileList.dirs()
    for path_task in path_task_list:
        path_task_images = os.path.join(path_task, 'images')
        path_task_labels = os.path.join(path_task, 'labels')
        task_images = FileList(path_task_images).images()
        task_labels = FileList(path_task_labels).files()
        copy_files(task_images, path_images)
        copy_files(task_labels, path_labels)


def main(args):
    if not args.merge:
        init_tasks(args.input, args.output, args.number)
    else:
        merge_tasks(args.input, args.output)


if __name__ == '__main__':
    args = get_args()
    print_args(args)
    main(args)
