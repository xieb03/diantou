import os
import random
import time
from random import shuffle

import numpy as np
import tiktoken
from openai import OpenAI


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


# 一个封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果
def get_chat_completion_content(client: OpenAI, user_prompt, system_prompt=None, model="gpt-3.5-turbo", temperature=0.2,
                                print_token_count=True):
    start_time = time.time()
    token_count = 0
    encoding = tiktoken.encoding_for_model(model)
    messages = list()
    if system_prompt is not None:
        messages.append(dict(role="system", content=system_prompt))
        token_count += len(encoding.encode(system_prompt))
    messages.append(dict(role="user", content=user_prompt))
    token_count += len(encoding.encode(user_prompt))

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # 模型输出的温度系数，控制输出的随机程度
    )

    # 调用 OpenAI 的 ChatCompletion 接口
    content = response.choices[0].message.content
    token_count += len(encoding.encode(content))

    end_time = time.time()
    cost_time = end_time - start_time
    if print_token_count:
        print(F"token count = {token_count}, cost time = {cost_time:.1F}s.")
    else:
        print(F"cost time = {cost_time:.1F}s.")

    return content


def main():
    pass


if __name__ == '__main__':
    main()
