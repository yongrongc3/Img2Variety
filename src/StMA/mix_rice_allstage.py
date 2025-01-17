import os
import cv2
import numpy as np
from datetime import datetime

# 定义混合增强函数
def mixgen(image1, image2, lam=0.5):
    return lam * image1 + (1 - lam) * image2

# 设置根文件夹路径
root_folder = '/93rice_twoviews_mixallstage/train'
print("Step1")
# 递归遍历文件夹，寻找图像文件
for subdir, _, files in os.walk(root_folder):
    # 过滤出符合条件的图像文件
    filtered_files = []
    # print(f"files:{files}")
    for file in files:
        
        if file.endswith('.png'):
            date_str = file.split('_')[1]
            file_date = datetime.strptime(date_str, '%Y-%m-%d')
            filtered_files.append(file)
    print("filtered_files")
    # 按前18个字符分组
    grouped_files = {}
    for file in filtered_files:
        key = file[:18]
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(file)
    
    # 对每组文件进行混合增强
    for key, file_list in grouped_files.items():
        if len(file_list) > 1:
            for i in range(len(file_list) - 1):
                file1_path = os.path.join(subdir, file_list[i])
                file2_path = os.path.join(subdir, file_list[i + 1])
                
                image1 = cv2.imread(file1_path)
                image2 = cv2.imread(file2_path)
                
                # 进行混合增强
                mixed_image = mixgen(image1, image2)
                
                # 保存混合后的图像
                mixed_filename = f"{file_list[i].split('.')[0]}_{file_list[i + 1].split('.')[0]}.png"
                mixed_file_path = os.path.join(subdir, mixed_filename)
                cv2.imwrite(mixed_file_path, mixed_image)
                print(f"mixed_filename：{mixed_filename}")
