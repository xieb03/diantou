import os.path

from util_date import *


# 列出所有目录下对应的文件的绝对路径，但不包括文件夹
def list_all_file_list(_folder_path_list: Union[str, List[str]]):
    file_path_list = list()
    for folder_path in to_list(_folder_path_list):
        # 注意 dirs 与 files 是并列的，即 root 下面包含 dirs（文件夹）和 files（文件）
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_path_list.append(file_path)

    return file_path_list


# 一个路径是否存在，同时支持文件或者文件夹
def is_file_or_dir_exist(_file_path):
    return os.path.exists(_file_path)


# 一个文件是否存在
def is_file_exist(_file_path):
    return is_file_or_dir_exist(_file_path) and is_file(_file_path)


# 一个文件夹是否存在
def is_dir_exist(_file_path):
    return is_file_or_dir_exist(_file_path) and is_dir(_file_path)


# 一个路径是否是文件
def is_file(_file_path):
    return os.path.isfile(_file_path)


# 一个路径是否是文件夹
def is_dir(_file_path):
    return os.path.isdir(_file_path)


# 获取文件的大小
def get_file_size(_file_path):
    assert is_file_or_dir_exist(_file_path), F"'{_file_path}' 不存在"
    assert is_file(_file_path), F"'{_file_path}' 不是一个文件"
    return os.path.getsize(_file_path)


# 获取文件夹的大小
def get_dir_size(_file_path):
    assert is_file_or_dir_exist(_file_path), F"'{_file_path}' 不存在"
    assert is_dir(_file_path), F"'{_file_path}' 不是一个文件夹"
    total_size = 0
    for item in os.listdir(_file_path):
        item_path = os.path.join(_file_path, item)
        if os.path.isfile(item_path):
            total_size += os.path.getsize(item_path)
    return total_size


# 删除文件路径后面的分隔符，确保后面连接的时候保持干净
def delete_end_path_separator(_path: str):
    while _path.endswith(PATH_SEPARATOR):
        _path = _path[:-1]

    return _path


# 确保结尾有且只有一个分隔符
def keep_one_end_path_separator(_path: str):
    return delete_end_path_separator(_path) + PATH_SEPARATOR


def main():
    assert not is_file_or_dir_exist(get_current_timestamp())
    # D:\PycharmProjects\xiebo\diantou\util_path.py
    # print(__file__)
    # 1360
    print(get_file_size(__file__))
    # 3677817
    print(get_dir_size(os.path.dirname(__file__)))


if __name__ == '__main__':
    main()
