# noinspection PyUnresolvedReferences
import atexit
import json
import math
import os
# noinspection PyUnresolvedReferences
import queue
import random
import re
import time
import traceback
# 可以看到，最终调用函数example时，是经过 @my_decorator装饰的，装饰器的作用是接受一个被包裹的函数作为参数，对其进行加工，
# 返回一个包裹函数，代码使用 @functools.wraps装饰将要返回的包裹函数wrapper，使得它的 __name__， __module__，和 __doc__
# 属性与被装饰函数example完全相同，这样虽然最终调用的是经过装饰的example函数，但是某些属性还是得到维护。
from functools import wraps
from random import shuffle
# noinspection PyUnresolvedReferences
from typing import List

import numpy as np
from IPython.display import Image, display, HTML

PATH_SEPARATOR = os.path.sep
BIGDATA_PATH = "D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\"
BIGDATA_IMAGE_PATH = BIGDATA_PATH + "images" + PATH_SEPARATOR
BIGDATA_WHISPER_PATH = BIGDATA_PATH + "whisper" + PATH_SEPARATOR
BIGDATA_VOICES_PATH = BIGDATA_PATH + "voices" + PATH_SEPARATOR


# 删除所有空字符串
def delete_all_blank(_str):
    return re.sub("\\s+", "", _str)


# 播放程序结束音乐
def play_music(_file_name=os.path.join(os.path.split(os.path.realpath(__file__))[0], "end_music.mp3")):
    os.system("open {}".format(_file_name))


# 用装饰器实现函数计时
def func_timer(arg=True, play_end_music=False, ignore_keyboard_interrupt=True, logger_function=None):
    if arg:
        def _func_timer(_function):
            @wraps(_function)
            def function_timer(*args, _play_end_music=play_end_music, **kwargs):
                t0 = time.time()
                try:
                    result = _function(*args, **kwargs)
                # 如果是人为停止任务(触发KeyboardInterrupt)，则不会触发音乐
                # 注意raise e必须执行，即将异常，否则return result会被执行，local variable ‘result’ referenced before assignment
                except KeyboardInterrupt as e:
                    if ignore_keyboard_interrupt:
                        _play_end_music = False
                    raise e
                except Exception as e:
                    traceback.print_exc()
                    output = "'" + _function.__name__ + "' get something wrong: " + str(e)
                    if logger_function is not None:
                        logger_function(output)
                    else:
                        print(output)
                    raise e
                finally:
                    t1 = time.time()
                    output = "'" + _function.__name__ + "' spent {:.4f}s.".format(t1 - t0)
                    if logger_function is not None:
                        logger_function(output)
                    else:
                        print(output)
                    if _play_end_music:
                        play_music()
                return result

            return function_timer
    else:
        def _func_timer(_function):
            return _function
    return _func_timer


# 更舒服的打印 json
def print_json(_data, _indent=4, _sort_keys=False, _ensure_ascii=False):
    # 如果不是纯数据，例如 openAI 返回的结果，可以自己解析为 json data
    if hasattr(_data, 'model_dump_json'):
        _data = json.loads(_data.model_dump_json())
    print(json.dumps(_data, indent=_indent, sort_keys=_sort_keys, ensure_ascii=_ensure_ascii))


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


# 设置 cell 的宽度，对于较大的 dataframe 更清楚，调整到 100% 即可，用 37% 比较接近原始值，每个屏幕可能不太一样，因为分辨率不一样，需要实际试一下
# FString 或者 str.format 中，用 {{ 来转义 {，用 }} 来转义 }。
def set_jupyter_cell_width(_width=37):
    display(HTML(F"<style>.container {{ width:{_width}% !important; }}</style>"))


# 模拟 tail -f 一个文件
def tailf(_file_path, _interval_duration=1, _interval_line=0.1, _callback=print, _encoding='utf-8'):
    line_list = list()
    with open(_file_path, 'r', encoding=_encoding) as fp:
        # 先读到最后
        while fp.readline():
            pass
        # 开始监控
        print('#' * 80)
        print(F"正在监控 '{_file_path}'，按 Ctrl + C 停止.")
        print('#' * 80)
        while True:
            line = fp.readline()
            if line and len(line.strip()) != 0:
                line_list.append(line.strip())
                if len(line_list) == _interval_line:
                    if _callback is not None:
                        _callback(line_list)
                    line_list = list()
            else:
                print("sleep")
                time.sleep(_interval_duration)


def main():
    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == (1, 2, 3)).all()
    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == [1, 2, 3]).all()

    tailf(r"D:\PycharmProjects\xiebo\diantou\bigdata\temp1.txt")


if __name__ == '__main__':
    main()
