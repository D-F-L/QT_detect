import os
import shutil
import glob

# 文件夹路径
folder_A = "/data/sdv1/tianshengzhao/UnderwaterDataSet/DataSet-LC"  # 替换为文件夹A的路径
folder_B = "/data/sdv1/xiangrui/denoiser/data/noise/onc"  # 替换为文件夹B的路径

# 确保文件夹B存在
if not os.path.exists(folder_B):
    os.makedirs(folder_B)

# 获取文件夹A中的前500个文件名
files = glob.glob(os.path.join(folder_A, '*.txt'))  # 只读取.txt文件
files.sort()  # 按文件名排序
# 列表
indices = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 24, 25, 29, 30, 31, 35, 37, 38, 39, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 112, 113, 114, 115, 116, 117, 119, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 136, 138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 165, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 191, 192, 193, 194, 195, 197, 201, 202, 203, 204, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 222, 225, 226, 227, 228, 229, 230, 232, 233, 234, 235, 236, 239, 240, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 261, 262, 263, 264, 265, 266, 267, 268, 472, 475, 476, 478, 480, 481, 482, 483, 484, 485, 486, 490, 491, 492, 499]  # 示例列表，替换为你的实际列表
print(len(indices))
# 遍历列表中的每个元素
for index in indices:
    if 0 <= index < len(files):  # 确保索引有效
        # 获取文件夹A中对应的文件名
        file_name = files[index]
        # 构造源文件路径和目标文件路径
        source_path = file_name
        tmp = file_name.split('_')[-1]
        target_path = os.path.join(folder_B, tmp)
        # 复制文件
        shutil.copy2(source_path, target_path)
        print(f"Copied {file_name} from folder A to folder B")
    else:
        print(f"Invalid index: {index} (out of range)")