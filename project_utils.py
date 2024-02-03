import datetime
import math
import os
import random
import time
# noinspection PyUnresolvedReferences
from base64 import b64decode
from random import shuffle

import numpy as np
import requests
# noinspection PyUnresolvedReferences
import tiktoken
# noinspection PyUnresolvedReferences
import torch
# noinspection PyUnresolvedReferences
from IPython.display import IFrame, Image
from openai import OpenAI
# noinspection PyUnresolvedReferences
from torch import nn

PATH_SEPARATOR = os.path.sep
BIGDATA_PATH = "D:\\PycharmProjects\\xiebo\\diantou\\"
BIGDATA_IMAGE_PATH = BIGDATA_PATH + PATH_SEPARATOR + "images" + PATH_SEPARATOR


# 展示图片，主要用于 jupyter notebook
def show_image(_url_or_local_image_path, width=None, height=None):
    if os.path.exists(_url_or_local_image_path):
        return Image(filename=_url_or_local_image_path, width=width, height=height)
    else:
        return Image(url=_url_or_local_image_path, width=width, height=height)


# 将 datetime.date 或者datetime.datetime 类型转化为指定样式的字符串
def change_datetime_to_str(_datetime, _style="%Y-%m-%d-%H:%M:%S"):
    return _datetime.strftime(_style)


# 从时间戳获得datetime
def get_datetime_from_timestamp(_timestamp):
    if isinstance(_timestamp, (int, float, str, np.int64, np.int32)):
        try:
            _timestamp = int(_timestamp)
        except ValueError:
            raise ValueError(_timestamp)

        if len(str(_timestamp)) == 13:
            _timestamp = _timestamp // 1000
        if len(str(_timestamp)) != 10:
            raise ValueError(_timestamp)
    else:
        raise ValueError(type(_timestamp))

    return datetime.datetime.fromtimestamp(_timestamp)


# 获得get请求结果
def get_get_response_from_url(_url, _params=None, _timeout=120, _try_times=3, _print=False, **kwargs):
    if _print:
        print(_url)
    while _try_times >= 0:
        try:
            result = requests.get(url=_url, params=_params, timeout=_timeout, **kwargs)
            if result.status_code != 200:
                raise ValueError(F"{result.status_code}, 状态码错误.")
            return result
        except Exception as e:
            print(type(e))
            print(e)
            time.sleep(3)
            _try_times -= 1
            if _try_times > 0:
                print("最后第 {try_times} 次尝试".format(try_times=_try_times))
            else:
                raise e


# 获得向量 vector 的 p 范数，默认 p = 1
def get_vector_norm(_vector: torch.Tensor, _p=2):
    # 必须是 vector
    assert _vector.dim() == 1, _vector.dim
    return torch.linalg.vector_norm(_vector, ord=_p)


# 求正则化结果
def get_tensor_norm(_x: torch.Tensor, _dim, _keepdim=True, _unbiased=False, _eps=1E-5):
    return (_x - _x.mean(dim=_dim, keepdim=_keepdim)) / torch.sqrt(
        (_x.var(dim=_dim, keepdim=_keepdim, unbiased=_unbiased) + _eps))


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


# 获取一个 sample
def get_a_sample_tensor(_shape, _random=False, _dtype=torch.float):
    _shape = to_tuple(_shape)
    if _random:
        return torch.randn(size=_shape, dtype=_dtype)

    count = 1
    for value in _shape:
        count *= value
    return torch.arange(count, dtype=_dtype).reshape(_shape)


# 统一转化为 tensor 来比较
def assert_tensor_equal(a, b, _force=False):
    # torch.equal 的两个输入必须都是 Tensor，直接返回 True or False
    # To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
    # or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)
    # assert torch.equal(torch.tensor(a), torch.tensor(b)), F"{a} != {b}"
    x, y = a, b
    if not isinstance(a, torch.Tensor):
        x = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        y = torch.tensor(b)
    assert torch.equal(x, y), F"{a} != {b}"


# 统一转化为 tensor 来比较
def assert_tensor_close(a, b, rel_tol=1e-05, abs_tol=1e-08):
    # torch.isclose 的两个输入必须都是 Tensor，返回 dtype = bool 的 tensor
    x, y = a, b
    if not isinstance(a, torch.Tensor):
        x = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        y = torch.tensor(b)
    assert x.shape == y.shape, F"{x.shape} not close {y.shape}"
    assert torch.isclose(x, y, rel_tol, abs_tol).all(), (
        "{} not close {}, rel_err = {}, abs_err = {}".format(a, b, *get_tensor_rel_abs_error(a, b)))


# 获得 rel_error 和 abs_error，都是全局最大的，且取过绝对值
def get_tensor_rel_abs_error(a, b):
    assert_tensor_shape_equal(a, b)

    abs_error = torch.max(torch.abs(a - b))
    rel_error = torch.max(abs_error / torch.minimum(torch.abs(a), torch.abs(b)))
    return rel_error, abs_error


# 比较一个 tensor 的形状，注意，必须用 tuple 而不能是 list
# 如果是 torch.Size([])，需要传入 tuple() 进行比较，当然也可以传入 torch.Size([])
# 也可以传入两个 tensor 直接比较形状
def assert_tensor_shape_equal(_tensor: torch.Tensor, _shape_or_tensor):
    if isinstance(_shape_or_tensor, torch.Tensor):
        # 注意 shape 是 torch.Size 不是 torch.Shape
        assert_tensor_equal(_tensor.shape, _shape_or_tensor.shape)
    else:
        assert_tensor_equal(_tensor.shape, to_tuple(_shape_or_tensor))


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


def print_closeai():
    openai_base_url, openai_api_key = get_closeai_parameter()
    openai_api_key = shuffle_str(openai_api_key)
    print(F"OPENAI_BASE_URL: {openai_base_url}")
    print(F"OPENAI_API_KEY: {openai_api_key}")


def get_closeai_parameter():
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_base_url, openai_api_key


# 获得 openai 的 client，经过实验，并不是单例的
def get_openai_client():
    openai_base_url, openai_api_key = get_closeai_parameter()
    client = OpenAI(
        base_url=openai_base_url,
        api_key=openai_api_key,
    )
    return client


# user_prompt 和 system_prompt 都可以支持 list，但 system_prompt 一定都在 user_prompt 之前调用
def get_chat_completion_content(client: OpenAI, user_prompt, system_prompt=None, model="gpt-3.5-turbo", temperature=0.2,
                                print_token_count=True, print_response=False):
    start_time = time.time()
    # token_count = 0
    # encoding = tiktoken.encoding_for_model(model)

    messages = list()
    if system_prompt is not None:
        for prompt in to_list(system_prompt):
            messages.append(dict(role="system", content=prompt))
            # token_count += len(encoding.encode(prompt))

    for prompt in to_list(user_prompt):
        messages.append(dict(role="user", content=prompt))
        # token_count += len(encoding.encode(prompt))

    # 调用 OpenAI 的 ChatCompletion 接口
    # ChatCompletion(id='chatcmpl-NppKIdfzNiPgzQJRJDSQ1qa3240CP',
    # choices=[Choice(finish_reason='stop', index=0, logprobs=None,
    # message=ChatCompletionMessage(content='我是一个人工智能助手，没有性别和年龄。', role='assistant',
    # function_call=None, tool_calls=None))], created=1706940431, model='gpt-3.5-turbo-0613',
    # object='chat.completion', system_fingerprint=None,
    # usage=CompletionUsage(completion_tokens=20, prompt_tokens=26, total_tokens=46))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # 模型输出的温度系数，控制输出的随机程度
    )

    if print_response:
        print(response)

    choices = response.choices
    choice_count = len(choices)
    # 目前只取第一条
    choice = choices[0]
    content = choice.message.content
    finish_reason = choice.finish_reason
    # token_count += len(encoding.encode(content))

    prompt_token_count = response.usage.prompt_tokens
    completion_token_count = response.usage.completion_tokens
    total_token_count = prompt_token_count + completion_token_count

    end_time = time.time()
    cost_time = end_time - start_time
    if print_token_count:
        print(F"prompt_token_count = {prompt_token_count}, completion_token_count = {completion_token_count}, "
              F"total_token_count = {total_token_count}.")

    print(F"choice_count = {choice_count}, cost time = {cost_time:.1F}s, finish_reason = {finish_reason}.")

    return content


# 一个封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果
# # 图像创建接口只支持一个 prompt
def get_image_create(client: OpenAI, prompt, model="dall-e-3", response_format="b64_json", size="1024x1024",
                     style="natural", print_response=False, save_image=True, image_name=None):
    start_time = time.time()

    if len(prompt) >= 4000:
        raise ValueError(F"prompt 最多只支持 4000 字的 prompt，目前是 {len(prompt)}.")

    if style not in {"vivid", "natural"}:
        raise ValueError(F"style 只支持 vivid 和 b64_json，目前是 {style}.")

    if response_format not in {"b64_json", "url"}:
        raise ValueError(F"response_format 只支持 b64_json 和 url，目前是 {response_format}.")

    # noinspection PyTypeChecker
    # ImagesResponse(created=1706939997, data=[Image(b64_json=None,
    # revised_prompt='Display an affectionate and friendly scene between a cat and a dog.
    # This lovely image should portray the deep camaraderie shared by these two animals,
    # where the cat, a beautiful long-haired Persian with silky white fur and piercing green eyes,
    # brushes up against the dog, a playful golden retriever with a shiny blonde coat and bright, warm eyes.
    # The background is tranquil – a cozy living room setting with a fireplace gently burning,
    # providing the perfect space for this expression of animal friendship.',
    # url='https://oaidalleapiprodscus.blob.core.windows.net/private/org-8ibGDXyMylPHUbmU1ZNXgim3/user-FL5wft4rVgt42TRFDkpwPVEW/img-cMEmSlKB3RqsnOkL2aO7GipP.png?st=2024-02-03T04%3A59%3A57Z&se=2024-02-03T06%3A59%3A57Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-02-03T02%3A40%3A53Z&ske=2024-02-04T02%3A40%3A53Z&sks=b&skv=2021-08-06&sig=z0XR71uXPx7nn4U2I7tgAV8sn%2B6IZYgQSjSNjo4ZrFs%3D')])
    response = client.images.generate(
        model=model,
        prompt=prompt,
        quality="hd",
        response_format=response_format,
        size=size,
    )

    if print_response:
        print(response)

    create_timestamp = response.created
    create_datetime = change_datetime_to_str(get_datetime_from_timestamp(create_timestamp))

    data = response.data
    revised_prompt = data[0].revised_prompt
    b64_json = data[0].b64_json
    url = data[0].url

    end_time = time.time()
    cost_time = end_time - start_time

    print(F"cost time = {cost_time:.1F}s.")

    # 默认图片名：日期 + 提示词前面 20 个字符 + .png
    local_image_path = (BIGDATA_IMAGE_PATH
                        + (image_name if image_name is not None else create_datetime + "-" + prompt[:20]) + ".png")

    url_or_local_image_path = url
    if response_format == "url":
        # 如果保存图片的话，返回的就是本地路径，否则返回 url。外面可以统一用 show_image 来展示，已经做了兼容
        if save_image:
            res = get_get_response_from_url(url)
            with open(local_image_path, 'wb') as fp:
                fp.write(res.content)
            url_or_local_image_path = local_image_path
        return revised_prompt, url_or_local_image_path
    elif response_format == "b64_json":
        return revised_prompt, b64_json
    else:
        raise NotImplementedError(F"还没有实现 {response_format}.")


def main():
    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == (1, 2, 3)).all()
    # noinspection PyUnresolvedReferences
    assert (np.array([1, 2, 3]) == [1, 2, 3]).all()

    with get_openai_client() as client:
        # content = get_chat_completion_content(client,
        #                                       user_prompt=["你是男生女生", "你的年纪是多大？"],
        #                                       temperature=0.8, print_response=True)
        # print(content)

        revised_prompt, url = get_image_create(client, "展示一只猫和一只狗亲密友好的画面。", response_format="url",
                                               print_response=True)
        print(revised_prompt, url)
        revised_prompt, b64_json = get_image_create(client, "展示一只猫和一只狗亲密友好的画面。",
                                                    response_format="b64_json", print_response=True)
        print(revised_prompt, b64_json)


if __name__ == '__main__':
    main()
