import os
import pathlib
import shutil
from typing import Any, Callable, List
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


##########################
# utils
##########################

def apply_to_file(files: List[str], func: Callable) -> List[Any]:
    '''
    将单文件处理函数应用在多文件上
    @func: 文件处理函数
    @files: 文件路径列表
    @return: 列表 [func(file), ...]
    '''
    outputs = []
    for file in tqdm(files):
        outputs.append(func(file))
    return outputs


def apply_to_files(files: List[str], func: Callable, num_processes: int = 1) -> List[Any]:
    '''
    对多文件进程拆分, 多进程加速处理文件, 合并处理结果, 需在 if main 下执行
    @func: 多文件处理函数
    @files: 文件路径列表
    @return: 列表 [func(file), ...]
    '''
    assert 0 < num_processes <= cpu_count(), f'进程数{num_processes}不合法'
    # 将文件列表按照进程数有序分成多个子列表
    sublist = []
    batch_size = len(files) // num_processes if len(files) % num_processes == 0 else len(files) // num_processes + 1
    for i in range(num_processes):
        start = i * batch_size
        end = start + batch_size
        sublist.append(files[start : end])
    # 处理每个子列表的文件
    with Pool(num_processes) as p:
        p_outputs = p.map(func, sublist)
    # 合并所有子列表
    outputs = []
    for p_output in p_outputs:
        outputs.extend(p_output)
    return outputs


class FuncFactory:
    def __init__(self) -> None:
        '''
        函数工厂，注册函数名到函数的映射
        '''
        self._funcTable = dict()
    
    def register(self, *func_name_list: str):
        def wraper(func):
            for func_name in func_name_list:
                assert not func_name in self._funcTable.keys(), f"{func_name} already exists !!!"
                self._funcTable[func_name] = func
            return func
        return wraper

    def getFunction(self, func_name: str):
        return self._funcTable[func_name]


class FileList:
    def __init__(self, root: str) -> None:
        '''
        文件列表类, 按照指定条件过滤并列出root下的所以子项路径
        '''
        self._subs = self._list_root(root)

    def _list_root(self, root: str) -> List[str]:
        '''
        列出目录下所有文件
        '''
        assert os.path.isdir(root), f"{root} is not a dir"
        subs = []
        for subname in os.listdir(root):
            subpath = os.path.join(root, subname)
            subs.append(subpath)
        return subs

    def suffixes(self, *suffixes: str) -> List[str]:
        '''
        列出含有后缀的文件，无后缀则列出所有子项路径，如 .avi .jpg .txt等
        '''
        if len(suffixes) == 0:
            return self._subs
        files = []
        for sub in self._subs:
            if os.path.isfile(sub) and pathlib.Path(sub).suffix in suffixes:
                files.append(sub)
        return files

    def dirs(self) -> List[str]:
        '''
        列出所有子目录
        '''
        dirs = []
        for sub in self._subs:
            if os.path.isdir(sub):
                dirs.append(sub)
        return dirs

    def images(self) -> List[str]:
        '''
        列出所有图片
        '''
        return self.suffixes('.jpg', '.jpeg', '.png')

    def videos(self) -> List[str]:
        '''
        列出所有视频
        '''
        return self.suffixes('.avi', '.mp4')


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

