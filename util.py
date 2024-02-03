import math
import os
import random
import re
from random import shuffle
# noinspection PyUnresolvedReferences
from typing import List

import numpy as np
from IPython.display import Image

PATH_SEPARATOR = os.path.sep
BIGDATA_PATH = "D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\"
BIGDATA_IMAGE_PATH = BIGDATA_PATH + "images" + PATH_SEPARATOR


# 逐行打印 list
def print_list(_list, _pre="", _suf="", _with_index=True, _start_index=0):
    if _with_index:
        for index, value in enumerate(_list):
            print(F"{_pre}{index + _start_index}: {value}{_suf}")

    else:
        for value in _list:
            print(F"{_pre}{value}{_suf}")


# 展示图片，主要用于 jupyter notebook
# 兼容 图片数据 data、本地图片 filename、远程图片 url
def show_image(_url_or_local_image_path_or_data, width=None, height=None):
    if isinstance(_url_or_local_image_path_or_data, bytes):
        return Image(data=_url_or_local_image_path_or_data, width=width, height=height)
    elif os.path.exists(_url_or_local_image_path_or_data):
        return Image(filename=_url_or_local_image_path_or_data, width=width, height=height)
    else:
        return Image(url=_url_or_local_image_path_or_data, width=width, height=height)


# 返回一个 list 中部分索引构成的新 list
def get_list_from_index(_list, _index_list):
    result = list()
    for index in to_list(_index_list):
        result.append(_list[index])
    return result


# 返回一个 list 中排除索引构成的新索引
def get_index_exclude_index(_list, _index_list):
    _index_list = to_list(_index_list)
    length = len(_list)

    # 先将所有的 index 的可能负值取回正值
    index_list = list()
    for index in _index_list:
        if index < 0:
            index += length
        index_list.append(index)

    result = list()
    for index in range(length):
        if index not in index_list:
            result.append(index)
    return result


# 返回一个 list 中排除索引构成的新 list
def get_list_exclude_index(_list, _index_list):
    index_list = get_index_exclude_index(_list, _index_list)
    return get_list_from_index(_list, index_list)


# 如果不等就打印
def assert_equal(a, b):
    assert a == b, F"{a} != {b}"


def get_rel_abs_error(a, b):
    abs_error = abs(a - b)
    rel_error = abs_error / min(abs(a), abs(b))
    return rel_error, abs_error


def assert_close(a, b, rel_tol=1e-05, abs_tol=1e-08):
    assert math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol), (
        "{} not close {}, rel_err = {}, abs_err = {}".format(a, b, *get_rel_abs_error(a, b)))


# 固定随机种子
# noinspection PyBroadException,PyUnresolvedReferences
def fix_all_seed(_seed=13, _print=True, _simple=True):
    random.seed(_seed)
    np.random.seed(_seed)
    if not _simple:
        try:
            import tensorflow as tf
            tf.set_random_seed(_seed)
        except Exception:
            if _print:
                print("tensorflow sed random seed fail.")
        try:
            import torch
            torch.manual_seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            if _print:
                print("pytorch sed random seed fail.")


# 打乱字符串
def shuffle_str(_str):
    str_list = list(_str)
    # 调用random模块的shuffle函数打乱列表
    shuffle(str_list)
    return ''.join(str_list)


# 判断是否可以迭代，可指定是否包含字符串
def is_sequence(_obj, _include_str=False):
    # 注意 python 2 中要多一个 unicode 的判断
    try:
        iter(_obj)
        len(_obj)
        if _include_str:
            return True
        else:
            return not isinstance(_obj, (str, bytes))
    except (TypeError, AttributeError):
        return False


# 转成 list
def to_list(_value, _include_str=False):
    if not is_sequence(_value, _include_str):
        return [_value]
    if isinstance(_value, list):
        return _value
    return list(_value)


def to_tuple(_value, _include_str=False):
    if not is_sequence(_value, _include_str):
        return (_value,)
    if isinstance(_value, tuple):
        return _value
    return tuple(_value)


# 删除一个 list 中的重复元素，剩下的元素保序
def delete_repeat_value_and_keep_order(_list):
    result_list = list()
    result_set = set()
    for value in _list:
        if value not in result_set:
            result_list.append(value)
            result_set.add(value)
    return result_list


# 获取一个 format 原型字符串中的变量名称
def get_format_string_name_list(_str, _clean_empty=True, _delete_repeat=True):
    if _clean_empty:
        _str = re.sub("{\\s*}", "", _str)

    # 需要加上非贪婪模式
    pattern = re.compile(".*?{\\s*(.+?)\\s*?}")

    result_list = re.findall(pattern, _str)
    if _delete_repeat:
        result_list = delete_repeat_value_and_keep_order(result_list)

    return result_list


# 填充 format 原型字符串，未指定的的用空值代替
def fill_in_f_string(_f_string: str, _print=True, **kwargs):
    format_name_list = get_format_string_name_list(_f_string)
    format_name_dict = dict.fromkeys(format_name_list, "")

    if _print:
        for key, value in kwargs.items():
            if key not in format_name_dict:
                print(F"{key} = {value} not in original string.")

    format_name_dict.update(kwargs)
    return _f_string.format(**format_name_dict)


def main():
    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == (1, 2, 3)).all()
    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == [1, 2, 3]).all()


if __name__ == '__main__':
    main()
