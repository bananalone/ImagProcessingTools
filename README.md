# ImagProcessingTools
图像批量处理工具  
1. 图像去重

## 安装
### 1.创建虚拟环境
python3 -m venv /path/to/new/virtual/environment
### 2.激活虚拟环境
source /path/to/new/virtual/environment/bin/activate
### 3.安装依赖
pip install -r requirements.txt

## 使用
### 1.图像去重
usage: ImagDeduplication.py [-h] -r ROOT [-t TYPE] [-s SIMI] [-d] [-m]
                            [--dest DEST] [--num_processes NUM_PROCESSES]  

optional arguments:  
  -h, --help            show this help message and exit  
  -r ROOT, --root ROOT  path to images, required  
  -t TYPE, --type TYPE  select method in [dhash, ahash, phash, ssim], default
                        dhash  
  -s SIMI, --simi SIMI  Similarity to deduplication, default 0.8  
  -d, --delete          delete or not  
  -m, --move            move or not  
  --dest DEST           path to move to, default none  
  --num_processes NUM_PROCESSES
                        number of processes, multi process acceleration,
                        default 1  