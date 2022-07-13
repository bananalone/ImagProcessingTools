import os
import pathlib
import shutil
from typing import Any, Dict, List

from tqdm import tqdm


##########################
# utils
##########################

def apply_to_files(*suffixes: str):
    '''
    将对单文件的操作应用在多文件上
    @filter_suffixes: 过滤，只应用于含有该后缀的文件，如 .avi .jpg .txt等
    @return: 字典 {文件路径：返回值}
    '''
    def wraper(func):
        def inner(files: List[str]) -> Dict[str, Any]:
            rets = dict()
            for file in tqdm(files):
                if pathlib.Path(file).suffix in suffixes:
                    ret = func(file)
                    rets[file] = ret
            return rets
        return inner
    return wraper


class FuncFactory:
    def __init__(self) -> None:
        '''
        函数工厂，注册函数名到函数的映射
        '''
        self._funcTable = dict()
    
    def register(self, func_name: str):
        def wraper(func):
            assert not func_name in self._funcTable.keys(), f"{func_name} already exists !!!"
            self._funcTable[func_name] = func
            return func
        return wraper

    def getFunction(self, func_name: str):
        return self._funcTable[func_name]


def list_files_from_root(root: str):
    '''
    列出目录下所有子文件
    '''
    assert os.path.isdir(root), f"{root} is not a dir"
    files = []
    for file_name in os.listdir(root):
        file_path = os.path.join(root, file_name)
        if os.path.isfile(file_path):
            files.append(file_path)
    return files


##########################
# 文件操作
##########################

def remove_files(files: List[str]):
    '''
    删除列表里的文件
    '''
    for file in tqdm(files):
        if os.path.isfile(file):
            os.remove(file)


def move_files(files: List[str], dest_dir: str):
    '''
    移动列表里的文件到指定目录
    '''
    for file in tqdm(files):
        if os.path.isfile(file):
            shutil.move(file, os.path.join(dest_dir, os.path.basename(file)))

