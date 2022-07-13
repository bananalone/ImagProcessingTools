import argparse
import os
from typing import List, Dict, Set

import numpy as np
import cv2
from tqdm import tqdm

from common import apply_to_files, move_files, remove_files, list_files_from_root, FuncFactory


fFactory = FuncFactory()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True, help='Path to images')
    parser.add_argument('-t', '--type', type=str, default='dhash', help='select hash in [dhash, ahash], default use dhash')
    parser.add_argument('-s', '--simi', type=float, default=0.9, help='Similarity of de duplication')
    parser.add_argument('-d', '--delete', action='store_true', help='delete or not')
    parser.add_argument('-m', '--move', action='store_true', help='move or not')
    parser.add_argument('--dest', type=str, help='path to move')
    args = parser.parse_args()
    return args


@fFactory.register('dhash')
@apply_to_files('.jpg', '.jpeg')
def dhash(img_path) -> List[int]:
    '''
    计算图片的差异哈希值
    '''
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = 9, 8
    resize_gray = cv2.resize(gray, dsize=(w, h))
    hash = []
    for i in range(h):
        for j in range(w-1):
            if resize_gray[i, j] > resize_gray[i, j+1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash


@fFactory.register('ahash')
@apply_to_files('.jpg', '.jpeg')
def ahash(img_path) -> List[int]:
    '''
    计算图片的均值哈希值
    '''
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = 8, 8
    resize_gray = cv2.resize(gray, dsize=(w, h))
    avg = resize_gray.mean()
    hash = []
    for i in range(h):
        for j in range(w):
            if resize_gray[i, j] > avg:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def hamming_similarity(hash1, hash2) -> float:
    '''
    计算汉明距离，并转化为相似度
    '''
    assert len(hash1) == len(hash2), 'len(hash1) != len(hash2)'
    simi = 0
    for i in range(len(hash1)):
        if hash1[i] == hash2[i]:
            simi += 1
    return simi / len(hash1)


def find_dupl_imgs_from_hashtable(hashtable: Dict[str, float], simi: float) -> Set[str]:
    '''
    从图像哈希表中找出重复的图像
    '''
    img_paths = list(hashtable)
    assert len(img_paths) > 1, '至少两幅图像'
    dupl_imgs = set() # 重复图像集合
    for i in tqdm(range(len(img_paths)-1)):
        for j in range(i+1, len(img_paths)):
            hamm_simi = hamming_similarity(hashtable[img_paths[i]], hashtable[img_paths[j]])
            if hamm_simi > simi:
                dupl_imgs.add(img_paths[j])
    return dupl_imgs


def main():
    args = get_args()
    print(f'哈希类型: {args.type}')
    print(f"相似度阈值: {args.simi}")
    print(f'源目录: {args.root}')
    print(f'是否删除: {args.delete}')
    print(f'是否移动: {args.move}')
    print(f'移动到目录: {args.dest}')
    print('计算图像差异哈希值...')
    files = list_files_from_root(args.root)
    hash = fFactory.getFunction(args.type)
    hashtable = hash(files)
    print(f"共{len(hashtable)}幅图像")
    print('寻找重复图像...')
    dupl_imgs = find_dupl_imgs_from_hashtable(hashtable, simi=args.simi)
    print(f"共找出{len(dupl_imgs)}幅重复图像")
    if args.move:
        print("移动重复图像...")
        if not os.path.exists(args.dest):
            os.makedirs(args.dest)
        move_files(dupl_imgs, args.dest)
    elif args.delete:
        print("删除重复图像...")
        remove_files(dupl_imgs)


if __name__ == '__main__':
    main()