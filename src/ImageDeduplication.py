import argparse
import os
from typing import List, Callable

import numpy as np
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm

from common import apply_to_file, apply_to_files, move_files, print_args, remove_files, FuncFactory, FileList


DHASH = 'dhash'
AHASH = 'ahash'
PHASH = 'phash'
STRUCTURAL_SIMILARITY = 'ssim'
### more method ###


hashFactory = FuncFactory()
simiFactory = FuncFactory()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True, help='path to images, required')
    parser.add_argument('-t', '--type', type=str, default='dhash', help='select method in [dhash, ahash, phash, ssim], default dhash')
    parser.add_argument('-s', '--simi', type=float, default=0.8, help='Similarity to deduplication, default 0.8')
    parser.add_argument('-d', '--delete', action='store_true', help='delete or not')
    parser.add_argument('-m', '--move', action='store_true', help='move or not')
    parser.add_argument('--dest', type=str, default=None, help='path to move to, default none')
    parser.add_argument('--num_processes', type=int, default=1, help='number of processes, multi process acceleration, default 1')
    ### more arguments ###
    args = parser.parse_args()
    return args


@hashFactory.register(DHASH)
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


@hashFactory.register(AHASH)
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


@hashFactory.register(PHASH)
def phash(img_path) -> List[int]:
    '''
    计算图像的感知哈希值
    '''
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = 32, 32
    resize_gray = cv2.resize(gray, dsize=(w, h))
    dct = cv2.dct(np.float32(resize_gray))
    dct_roi = dct[0:8, 0:8]        
    avreage = np.mean(dct_roi)
    _phash = (dct_roi>avreage)+0
    hash = _phash.reshape(1,-1)[0].tolist()
    return hash


@hashFactory.register(STRUCTURAL_SIMILARITY)
def gray_resize_hash(img_path) -> np.ndarray:
    '''
    计算图片的均值哈希值
    '''
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = 64, 64
    resize_gray = cv2.resize(gray, dsize=(w, h))
    hash = resize_gray
    return hash


@simiFactory.register(AHASH, DHASH, PHASH)
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


@simiFactory.register(STRUCTURAL_SIMILARITY)
def ssim_similarity(img1, img2) -> float:
    '''
    计算结构相似度
    '''
    return structural_similarity(img1, img2)


def find_dupl_imgs(imgs: List[str], hash: Callable, simi: float, similarity: Callable) -> List[str]:
    '''
    找出重复的图像
    '''
    assert len(imgs) > 1, "至少两幅图像"
    imglist = sorted(imgs)
    hashlist = apply_to_file(imglist, hash)
    dupl_imgs = [] # 重复图像集合
    with tqdm(total=len(imglist)-1) as pbar:
        while len(imglist) > 1:
            img0 = imglist.pop(0)
            hash0 = hashlist.pop(0)
            for i in range(len(imglist)):
                val_simi = similarity(hashlist[i], hash0)
                if val_simi > simi:
                    dupl_imgs.append(img0)
                    break
            pbar.update()
    return dupl_imgs


def main(find_dupl_imgs_func, args):
    fileList = FileList(args.root)
    images = fileList.images()
    print(f"共{len(images)}幅图像")
    print('寻找重复图像...')
    dupl_imgs = apply_to_files(images, func=find_dupl_imgs_func, num_processes=args.num_processes)
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
    args = get_args()
    print_args(args)

    hash = hashFactory.getFunction(args.type)
    similarity = simiFactory.getFunction(args.type)

    # find_dupl_imgs_func = lambda x:find_dupl_imgs(x, hash=hash, simi=args.simi, similarity=similarity)
    def find_dupl_imgs_func(images):
        return find_dupl_imgs(images, hash=hash, simi=args.simi, similarity=similarity)
    
    main(find_dupl_imgs_func, args)
