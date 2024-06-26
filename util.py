import ast
# noinspection PyUnresolvedReferences
import atexit
import inspect
# noinspection PyUnresolvedReferences
import itertools
import json
import math
import operator
import os
# noinspection PyUnresolvedReferences
import queue
import random
import re
# noinspection PyUnresolvedReferences
import sys
import time
import traceback
# 要将一个字符串转换成对应的 python 类型可以使用 eval() 函数，可是这个函数有必定的安全漏洞，容易被不法人员利用，所以 python 中出了一个安全处理方式 ast.literal_eval()
# ast.literal_eval 仅能解析 Python 字面量和一些容器类型（如列表、字典、元组、集合等），这意味着它不能计算包含 Python 表达式或函数调用的字符串，例如 literal_eval("4 + 4") 会报错
# 这是一个设计上的限制，以确保安全性，防止执行潜在危险的代码。
# noinspection PyUnresolvedReferences
from ast import literal_eval
from collections import OrderedDict
# noinspection PyUnresolvedReferences
from collections import defaultdict
from functools import partial
# 可以看到，最终调用函数example时，是经过 @my_decorator装饰的，装饰器的作用是接受一个被包裹的函数作为参数，对其进行加工，
# 返回一个包裹函数，代码使用 @functools.wraps装饰将要返回的包裹函数wrapper，使得它的 __name__， __module__，和 __doc__
# 属性与被装饰函数example完全相同，这样虽然最终调用的是经过装饰的example函数，但是某些属性还是得到维护。
from functools import wraps
from random import shuffle
# noinspection PyUnresolvedReferences
from typing import List, Tuple, Dict, Union, Any, Optional, Mapping

import numpy as np
# noinspection PyUnresolvedReferences
import pandas as pd
from IPython.display import Image, display, HTML
# noinspection PyUnresolvedReferences
from tqdm import tqdm
from watermark import watermark

PATH_SEPARATOR = os.path.sep
BIGDATA_PATH = "D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\"
BIGDATA_PDF_PATH = BIGDATA_PATH + "pdfs" + PATH_SEPARATOR
BIGDATA_MD_PATH = BIGDATA_PATH + "mds" + PATH_SEPARATOR
BIGDATA_IMAGE_PATH = BIGDATA_PATH + "images" + PATH_SEPARATOR
BIGDATA_WHISPER_PATH = BIGDATA_PATH + "whisper" + PATH_SEPARATOR
BIGDATA_VOICES_PATH = BIGDATA_PATH + "voices" + PATH_SEPARATOR
BIGDATA_MODELS_PATH = BIGDATA_PATH + "models" + PATH_SEPARATOR
BIGDATA_DATA_PATH = BIGDATA_PATH + "data" + PATH_SEPARATOR
SPACY_DATA_PATH = BIGDATA_DATA_PATH + "spacy" + PATH_SEPARATOR
TORCH_TEXT_DATA_PATH = BIGDATA_DATA_PATH + "torch_text" + PATH_SEPARATOR

# "torch_dtype": "float16"
CHATGLM3_6B_model_id = "ZhipuAI/chatglm3-6b"
CHATGLM3_6B_model_revision = "v1.0.0"
CHATGLM3_6B_model_dir = BIGDATA_MODELS_PATH + "ZhipuAI" + PATH_SEPARATOR + "chatglm3-6b"

# "torch_dtype": "bfloat16"
# https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/summary
GLM4_9B_CHAT_model_id = "ZhipuAI/glm-4-9b-chat"
GLM4_9B_CHAT_model_revision = "master"
GLM4_9B_CHAT_model_dir = BIGDATA_MODELS_PATH + "ZhipuAI" + PATH_SEPARATOR + "glm-4-9b-chat"

# "torch_dtype": "float32"
BGE_LARGE_CN_model_id = "AI-ModelScope/bge-large-zh-v1.5"
BGE_LARGE_CN_model_revision = "master"
BGE_LARGE_CN_model_dir = BIGDATA_MODELS_PATH + "AI-ModelScope" + PATH_SEPARATOR + "bge-large-zh-v1___5"

# "torch_dtype": "float32"
BGE_LARGE_EN_model_id = "AI-ModelScope/bge-large-en-v1.5"
BGE_LARGE_EN_model_revision = "master"
BGE_LARGE_EN_model_dir = BIGDATA_MODELS_PATH + "AI-ModelScope" + PATH_SEPARATOR + "bge-large-en-v1___5"

# "torch_dtype": "float32"
BGE_RERANKER_LARGE_model_id = "quietnight/bge-reranker-large"
BGE_RERANKER_LARGE_revision = "master"
BGE_RERANKER_LARGE_model_dir = BIGDATA_MODELS_PATH + "quietnight" + PATH_SEPARATOR + "bge-reranker-large"

# "torch_dtype": "bfloat16"
# https://modelscope.cn/models/qwen/Qwen2-7B-Instruct/summary
QWEN2_7B_INSTRUCT_model_id = "qwen/Qwen2-7B-Instruct"
QWEN2_7B_INSTRUCT_model_revision = "master"
QWEN2_7B_INSTRUCT_model_dir = BIGDATA_MODELS_PATH + "qwen" + PATH_SEPARATOR + "Qwen2-7B-Instruct"

CHROMADB_PATH = BIGDATA_PATH + "chromadb" + PATH_SEPARATOR

CODE_BLOCK_REGEX = re.compile(r"```(.*?)```", re.DOTALL)
PYTHON_CODE_BLOCK_REGEX = re.compile(r"```python(.*?)```", re.DOTALL)
JSON_CODE_BLOCK_REGEX = re.compile(r"```json(.*?)```", re.DOTALL)

# 汉字中的数字转阿拉伯数字
CN_TO_NUMBER_DICT = OrderedDict()
for first_cn, first_number in zip(["十", "二十", "三十", "四十", "五十", "六十", "七十", "八十", "九十", ""],
                                  "123456789 "):
    for second_cn, second_number in zip(["一", "二", "三", "四", "五", "六", "七", "八", "九", ""], "1234567890"):
        cn = first_cn.strip() + second_cn.strip()
        number_str = str(first_number).strip() + str(second_number).strip()
        if len(cn) > 0:
            CN_TO_NUMBER_DICT[cn] = number_str

assert len(CN_TO_NUMBER_DICT) == 99, len(CN_TO_NUMBER_DICT)
assert len(CN_TO_NUMBER_DICT.values()) == 99
assert sum(map(int, CN_TO_NUMBER_DICT.values())) == 4950


# 按照字符字典替换字符串，注意是字符字典，而不是字符串字典，这样与顺序无关
def replace_chars(_str, _char_map):
    return ''.join(_char_map.get(c, c) for c in _str)


# 按照字符串字典替换字符串，注意是字符串字典，这样与顺序有关，最好传入 OrderedDict，注意与 replace_chars 的区别
def replace_strs(_str, _str_map):
    for k, v in _str_map.items():
        _str = _str.replace(k, v)
    return _str


# 打印历史对话
def print_history_message_list(history_message_list, _with_index=True, _start_index=1):
    index = _start_index
    for message in history_message_list:
        role = message["role"]
        content = message["content"].strip()
        if _with_index:
            print(F"{index} {role}:")
        else:
            print(F"{role}:")
        content = content.replace("\n", "\n\t")
        print(F'\t{content}')

        # 有 assistant 意味着要开始一轮的对话
        if role == "assistant":
            index += 1


# 检查 python 代码是否有语法错误，ast.parse 是用来解析语法树，如果语法有问题会报错
# noinspection PyBroadException
def check_python_code_syntax_error(_python_code, _print_error=False):
    try:
        ast.parse(_python_code)
        return True
    except Exception:
        if _print_error:
            traceback.print_exc()
        return False


# 从一段字符串中接触 三引号 的部分，主要用来抽离出 python 代码
# 支持多个三引号，会将多段代码串联起来
def extract_python_code(_response):
    if "```python" in _response:
        code_blocks = PYTHON_CODE_BLOCK_REGEX.findall(_response)
    else:
        code_blocks = CODE_BLOCK_REGEX.findall(_response)
    if code_blocks:
        if len(code_blocks) > 1:
            print(F"共发现 {len(code_blocks)} 段代码，会将它们串联起来.")
            # code_blocks = [code[7:] if code.startswith("python") else code for code in code_blocks]
            full_code = "\n".join(code_blocks)
            return full_code
        else:
            return code_blocks[0]
    else:
        print(_response)
        return None


class Colors:  # You may need to change color settings
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    ENDC = "\033[m"


# 将字典排序，注意这个并不是直接对 key 排序（字典是无序的），而是按照某一个 key 的 value_list 排序，同时其它 value_list 跟着变动
# {'names': ["a", "b", "c"], 'score': [2, 3, 1]} -> {'names': ["c", "a", "b"], 'score': [1, 2, 3]}
# 注意是原位修改
def sort_dict_with_one_value_list(_dict, _key, _sub_key=None, _reverse=False):
    key_index_dict = {_key: 0}
    total_list = [_dict[_key]]
    start = 1

    if _sub_key is not None:
        assert _sub_key != _key
        key_index_dict[_sub_key] = 1
        total_list.append(_dict[_sub_key])
        start = 2

    for key in _dict:
        if key == _key:
            continue
        if _sub_key is not None and key == _sub_key:
            continue
        key_index_dict[key] = start
        start += 1
        total_list.append(_dict[key])

    # zip 化
    total_list = list(zip(*total_list))
    if _sub_key is None:
        total_list.sort(key=operator.itemgetter(0), reverse=_reverse)
    else:
        total_list.sort(key=lambda item: (item[0], item[1]), reverse=_reverse)

    # 还原
    total_list = list(zip(*total_list))
    for key in _dict:
        _dict[key] = list(total_list[key_index_dict[key]])

    return _dict


# 保留几位有效数字
def get_round_value(_value, _digit):
    return round(_value, _digit)


get_round_6 = partial(get_round_value, _digit=6)
get_round_5 = partial(get_round_value, _digit=5)
get_round_4 = partial(get_round_value, _digit=4)
get_round_3 = partial(get_round_value, _digit=3)
get_round_2 = partial(get_round_value, _digit=2)
get_round_1 = partial(get_round_value, _digit=1)


# 返回字符串列表的所有字符数
def get_str_list_length(_str_list: List[str]) -> int:
    return sum(len(text) for text in _str_list)


# 返回 document 列表的每一个的字符数
def get_str_list_each_length(_str_list: List[str]) -> List[int]:
    return [len(text) for text in _str_list]


# 返回一个函数，其作用是获得一个对象的某个属性，注意是属性而不是方法
# 如果不包含这个属性，会抛异常：AttributeError: '...' object has no attribute '...'.
def function_get_attr(_attr):
    return lambda x: getattr(x, _attr)


# 获得一个函数的源代码
def get_function_source(_func):
    return inspect.getsource(_func)


# 打印一个对象的所有方法，ignore 表示去掉魔术方法
def print_dir(_x, _ignore=True):
    dirs = dir(_x)
    if _ignore:
        dirs = [_dir for _dir in dirs if not _dir.startswith("_") and not _dir.endswith("__")]
    print(dirs)


# 多个分隔符分割
# 注意 \\ 要多加一层，变成 \\
def multiply_split(_sep_list, _str):
    _sep_list = to_list(_sep_list)
    if "\\" in _sep_list:
        _sep_list[_sep_list.index("\\")] = "\\\\"
    return re.split("|".join(_sep_list), _str)


# 打印系统信息
def print_requirements():
    # deepspeed 几乎不能在 windows 下面安装
    packages = (
        "pip,"
        # pytorch
        "torch,torchvision,torchaudio,torchdata,torchtext,"
        # openai
        "openai,tiktoken,"
        # huggingface
        "transformers,tokenizers,huggingface_hub,"
        # modelscope
        "modelscope,"
        "deepspeed,peft,rouge_chinese,mpi4py,datasets,jieba,ruamel_yaml,networkx,brotlipy,brotli,"
        # data science
        "scikit-learn,numpy,pandas,matplotlib,seaborn,scipy,"
        # nltk
        "nltk,"
        # langchain，注意有的时候 package 中是 -，但是 import 要用 _
        "langchain,langchain_core,langchain_openai,langchain_community,langsmith,langgraph,pymupdf,rapidocr_onnxruntime,unstructured,"
        # web
        "flask,streamlit,"
        # jupyter
        "jupyter_server,jupyterlab,"
        # spider
        "scrapy,selenium,urllib3,requests,"
        # logger
        "loguru,"
        # database
        "pymongo,redis,pymysql,sqlalchemy,chromadb"
    )
    print(
        watermark(updated=True, current_date=True, current_time=True, timezone=True, python=True, conda=True,
                  hostname=True,
                  machine=True, githash=False, gitrepo=False, gitbranch=False, watermark=False, gpu=True,
                  packages=packages))


# 删除所有空字符串
def delete_all_blank(_str):
    return re.sub("\\s+", "", _str)


delete_all_blanks = delete_all_blank


# 删除所有回车，注意包含统一回车为 \n
def delete_all_break_lines(_str):
    return re.sub(r'\n+', '', replace_break_line(_str))


# 删除所有空格
def delete_all_spaces(_str):
    return re.sub(r' +', '', _str)


# 将多个空格变为一个
def replace_multiple_spaces(_str):
    return re.sub(r' +', ' ', _str)


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


# 逐行打印 dict
def print_dict(_dict, _pre="", _suf="", _with_index=False, _start_index=0):
    if _with_index:
        for index, (key, value) in enumerate(_dict.items()):
            print(F"{_pre}{index + _start_index}: {key}: {value}{_suf}")

    else:
        for (key, value) in _dict.items():
            print(F"{_pre}{key}: {value}{_suf}")


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
# 注意要想保证可以复现，必须整个过程也是一致的，因为本质上是每次需要随机的时候，新种子的序列都是一致的。如果过程发生变化，例如中间插入了某个随机过程，那么后面的新种子相当于有延迟，就不能保证可以复现了
# https://zhuanlan.zhihu.com/p/629526120
# noinspection PyBroadException,PyUnresolvedReferences
def fix_all_seed(_seed=13, _print=True, _simple=True, _warn_only=False):
    random.seed(_seed)
    np.random.seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)
    if not _simple:
        try:
            import torch
            torch.manual_seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
            # 默认是 False
            torch.backends.cudnn.deterministic = True
            # 当torch.backends.cudnn.benchmark 选项为 True 时候，cuda 为了提升训练效率，会自动试运行不同优化的卷积算法，
            # 以搜索最优最快的算法实现，由于不同硬件不同以及不同的版本的卷积算法实现，可能会导致训练结果不一致。
            # 所以，为了算法可复现，通常设置cudnn.benchmark = False。
            # 那什么情况可以设置 True:
            # 不考虑可复现性，当模型的输入和结构在训练过程保持固定不变化的时候，可以实现算法加速。
            # 否则，会因为反复的算法最优搜索导致额外的时间浪费。
            # 默认是 False
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True) 允许你配置 PyTorch 在可用的情况下使用确定性算法而不是非确定性算法，
            # 并且如果已知某个操作是不确定的(并且没有确定的替代方法)，则抛出 RuntimeError 错误。
            # warn_only (:class:`bool`, optional): If True, operations that do not have a deterministic implementation will throw a warning instead of an error. Default: ``False``
            torch.use_deterministic_algorithms(True, warn_only=_warn_only)
            # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)`
            # or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic
            # because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case,
            # you must set an environment variable before running your PyTorch application:
            # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
            # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            # torch.backends.cudnn.enabled 是 PyTorch 中的一个选项，用于指示是否启用 cuDNN（CUDA Deep Neural Network library）作为深度学习算法的加速库。
            # cuDNN 是由 NVIDIA 开发的针对深度神经网络进行优化的库，它使用了高度优化的算法和数据结构，可以显著提高深度学习模型的训练和推理速度。在使用 cuDNN 之前，我们需要确保 PyTorch 和 CUDA 都已正确配置。
            # 当 torch.backends.cudnn.enabled = True 时，PyTorch 会尝试使用 cuDNN 进行加速。cuDNN 会根据当前运行的 GPU 驱动版本、CUDA 版本和硬件设备的兼容性自动选择合适的算法，并将其作为底层运算库来加速 PyTorch 中的卷积、池化等操作。这样，深度学习模型的训练和推理速度可以得到显著提升。
            # 然而，有时我们可能需要禁用 cuDNN 加速。虽然 cuDNN 对大多数情况都是有益的，但在某些特定情况下，其使用可能导致不稳定的结果或不同于其他加速库的行为。因此，当 torch.backends.cudnn.enabled = False 时，PyTorch 将不会尝试使用 cuDNN 进行加速，而是使用纯 Python 实现的算法。这可能会带来一定的性能损失，但有时可以避免不稳定的行为。
            # 总之，torch.backends.cudnn.enabled 是一个控制是否启用 cuDNN 加速的选项。根据具体情况，我们可以根据需求选择启用或禁用 cuDNN。
            # 我们可以根据是否需要确定性的结果来选择是否打开这一选项，建议显检验是否具有确定性，如果前面的设定不足以维持确定性，可以把这一项置为 False，因为可能根本就没用到 cuDNN 的加速，所以这一项等于没有用处
            # 默认是 True
            # torch.backends.cudnn.enabled = False
        except Exception:
            if _print:
                print("pytorch sed random seed fail.")
        try:
            import tensorflow as tf
            tf.set_random_seed(_seed)
        except Exception:
            if _print:
                print("tensorflow sed random seed fail.")


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


# list 是不是没有重复元素
def is_list_unique(_list):
    return len(set(_list)) == len(_list)


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


# 获取一个字符的Unicode编码（即其的\u形式）
def get_unicode_escape(_char):
    code_point = ord(_char)
    return r'\u{:04x}'.format(code_point)


# # 而关于中文的正则表达式, 应该是 [\u4E00-\u9FFF], 和论坛里常被人提起的^[\u4E00-\u9FA5]很接近
# # 需要注意的是论坛里说的 [\u4E00-\u9FA5] 这是专门用于匹配简体中文的正则表达式
# # _include_traditional 表示是否包含繁体字
# # 实测不太对，例如 華
# def check_char_chinese(_char, _include_traditional=True):
#     if _include_traditional:
#         return '\u4e00' <= _char <= '\u9fff'
#     else:
#         return '\u4e00' <= _char <= '\u9fa5'


def check_char_chinese(_char):
    return '\u4e00' <= _char <= '\u9fff'


# 统一换行符为 \n
# 微软Windows系统：每行结尾有“<回车><换行>”，即“\r\n”
# Unix/Linux系统: 每行结尾只有“<换行>”，即"\n"
# 苹果Mac系统： 每行结尾只有“<回车>”，即"\r"
def replace_break_line(_str):
    return _str.replace("\r\n", "\n").replace("\r", "\n")


# 删除额外的回车，langchain 会在原本两个符号中间插入了 \n
# 注意这里只匹配一个
# 。\n“...”\n。 -> 。“...”。
def delete_one_extra_break_line(_str):
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    return re.sub(pattern, lambda match: match.group(0).replace('\n', ''), _str)


# 删除所有汉字间的多余空格，但保留英文字符间的，例如 "how are you", "one bag", "1 bag"
def delete_all_spaces_between_chinese(_str):
    pattern = re.compile(r'[\u4e00-\u9fff]( +)[\u4e00-\u9fff]', re.DOTALL)
    return re.sub(pattern, lambda match: delete_all_spaces(match.group(0)), _str)


def main():
    # Last updated: 2024-06-23 13:13:01中国标准时间
    #
    # Python implementation: CPython
    # Python version       : 3.11.5
    # IPython version      : 8.15.0
    #
    # pip                 : 23.2.1
    # torch               : 2.3.0+cu121
    # torchvision         : 0.18.0+cu121
    # torchaudio          : 2.3.0+cu121
    # torchdata           : 0.7.1
    # torchtext           : 0.18.0
    # openai              : 1.34.0
    # tiktoken            : 0.7.0
    # transformers        : 4.41.2
    # tokenizers          : 0.19.1
    # huggingface_hub     : 0.23.4
    # modelscope          : 1.15.0
    # deepspeed           : not installed
    # peft                : 0.7.1
    # rouge_chinese       : 1.0.3
    # mpi4py              : 3.1.5
    # datasets            : 2.18.0
    # jieba               : 0.42.1
    # ruamel_yaml         : 0.18.6
    # networkx            : 3.3
    # brotlipy            : not installed
    # brotli              : 1.1.0
    # scikit-learn        : 1.4.1.post1
    # numpy               : 1.24.3
    # pandas              : 2.2.1
    # matplotlib          : 3.8.3
    # seaborn             : 0.13.2
    # scipy               : 1.11.1
    # nltk                : 3.8.1
    # langchain           : 0.2.5
    # langchain_core      : 0.2.9
    # langchain_openai    : 0.1.8
    # langchain_community : 0.2.5
    # langsmith           : 0.1.81
    # langgraph           : 0.0.69
    # pymupdf             : 1.24.5
    # rapidocr_onnxruntime: 1.3.22
    # unstructured        : 0.14.6
    # flask               : 3.0.3
    # streamlit           : 1.36.0
    # jupyter_server      : 2.14.1
    # jupyterlab          : 4.2.2
    # scrapy              : 2.11.2
    # selenium            : 4.21.0
    # urllib3             : 2.2.2
    # requests            : 2.31.0
    # pymongo             : 4.7.3
    # redis               : 5.0.5
    # pymysql             : 1.1.1
    # sqlalchemy          : 1.4.39
    # chromadb            : 0.5.1
    #
    # conda environment: base
    #
    # Compiler    : MSC v.1916 64 bit (AMD64)
    # OS          : Windows
    # Release     : 10
    # Machine     : AMD64
    # Processor   : Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
    # CPU cores   : 32
    # Architecture: 64bit
    #
    # Hostname: SK-20231110MMDM
    #
    # GPU Info:
    #   GPU 0: NVIDIA GeForce RTX 4090
    print_requirements()

    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == (1, 2, 3)).all()
    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == [1, 2, 3]).all()

    # tailf(r"D:\PycharmProjects\xiebo\diantou\bigdata\temp1.txt")

    a = r"1,3\4,/5"
    assert_equal(multiply_split(",", a), ['1', '3\\4', '/5'])
    assert_equal(multiply_split([",", "\\", "/"], a), ['1', '3', '4', '', '5'])

    assert_equal(sort_dict_with_one_value_list({'names': ["a", "d", "c", "b"], 'score': [3, 2, 1, 2]}, "score"),
                 {'names': ['c', 'd', 'b', 'a'], 'score': [1, 2, 2, 3]})
    assert_equal(
        sort_dict_with_one_value_list({'names': ["a", "d", "c", "b"], 'score': [3, 2, 1, 2]}, "score", "names"),
        {'names': ['c', 'b', 'd', 'a'], 'score': [1, 2, 2, 3]})

    python_code = "blue_car_position = [current_position[0] + 10, current_position[1], current_position[2]]"
    assert check_python_code_syntax_error(python_code)

    python_code = "blue_car_position = [current_position[0] + 10, current_position[1], current_position[2]"
    assert not check_python_code_syntax_error(python_code)

    # 注意要用 r'\u4f60' 而不是 '\u4f60'，后者不是字符串，而是字符
    assert_equal(get_unicode_escape("华"), r'\u534e')
    assert_equal(get_unicode_escape("華"), r'\u83ef')
    # 注意汉语的标点符号不在汉字体系内
    assert_equal(get_unicode_escape("。"), r'\u3002')
    assert_equal(get_unicode_escape("."), r'\u002e')

    assert check_char_chinese("华")
    assert check_char_chinese("華")
    assert not check_char_chinese("。")
    assert not check_char_chinese("，")
    assert not check_char_chinese(".")

    assert_equal(delete_one_extra_break_line("。\n“...”\n。"), "。“...”。")
    assert_equal(delete_all_spaces_between_chinese("你好， 世  界！a b   c"), "你好， 世界！a b   c")
    assert_equal(replace_break_line("\r\n\r\n\r"), "\n\n\n")


if __name__ == '__main__':
    main()
