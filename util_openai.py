from base64 import b64decode
from threading import RLock

import tiktoken
import whisper
import zhconv
from openai import OpenAI

from util_spider import *


# 单例，实现获取 client
class Client(object):
    _single_lock = RLock()

    # 获得 openai 的 client，经过实验，并不是单例的
    @classmethod
    def _get_openai_client(cls) -> OpenAI:
        openai_base_url, openai_api_key = get_closeai_parameter()
        return OpenAI(
            base_url=openai_base_url,
            api_key=openai_api_key,
        )

    @classmethod
    def instance(cls) -> OpenAI:
        # 为了在多线程环境下保证数据安全，在需要并发枷锁的地方加上RLock锁
        with Client._single_lock:
            if not hasattr(Client, "_instance"):
                # print("第一次调用，申请一个 client")
                Client._instance = Client._get_openai_client()
                # 在程序结束使用的时候释放资源
                atexit.register(Client.exit)
            # 如果单例断开连接了，需要重新申请一个单例
            else:
                if Client._instance.is_closed():
                    # print("原来的 client 断开连接，重新申请了一个.")
                    Client._instance = Client._get_openai_client()
        return Client._instance

    @classmethod
    def exit(cls) -> None:
        # 为了在多线程环境下保证数据安全，在需要并发枷锁的地方加上RLock锁
        with Client._single_lock:
            if hasattr(Client, "_instance"):
                print("程序结束，close client.")
                Client._instance.close()
                Client._instance = None


def print_closeai():
    openai_base_url, openai_api_key = get_closeai_parameter()
    openai_api_key = shuffle_str(openai_api_key)
    print(F"OPENAI_BASE_URL: {openai_base_url}")
    print(F"OPENAI_API_KEY: {openai_api_key}")


def get_closeai_parameter():
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_base_url, openai_api_key


# 返回一个字符串的 token 数
def get_token_count(_str: str, _model="gpt-3.5-turbo", _encoding=None):
    if _encoding is None:
        _encoding = tiktoken.encoding_for_model(_model)
    return len(_encoding.encode(_str))


# user_prompt 和 system_prompt 都可以支持 list，但 system_prompt 一定都在 user_prompt 之前调用
# 历史消息队列
def get_chat_completion_content(user_prompt=None, system_prompt=None, messages=None,
                                model="gpt-3.5-turbo", temperature=0.1,
                                print_token_count=False, print_cost_time=False, print_response=False,
                                history_message_list: List = None,
                                using_history_message_list=True, tools=None, print_messages=False):
    start_time = time.time()
    # token_count = 0
    # encoding = tiktoken.encoding_for_model(model)

    client = Client.instance()

    if user_prompt is not None or system_prompt is not None:
        assert messages is None, "user_prompt 和 system_prompt 为一组，messages 为另外一组，这两组有且只能有一组不为空，目前前者不为空，但是后者也不为空."
    else:
        assert messages is not None, "user_prompt 和 system_prompt 为一组，messages 为另外一组，这两组有且只能有一组不为空，目前前者为空，后者也为空."

    total_messages = list()
    using_history = using_history_message_list and history_message_list is not None and len(history_message_list) != 0

    # 如果明确要求用历史对话，而且传入了历史对话，那么历史对话也要也要进入 prompt
    # 因为 closeAI 是负载均衡的，每次会随机请求不同的服务器，因此是不具备 openAI 自动保存一段历史对话的能力的，需要自己用历史对话来恢复
    if using_history:
        total_messages = history_message_list

    if messages is None:
        if system_prompt is not None:
            for prompt in to_list(system_prompt):
                total_messages.append(dict(role="system", content=prompt))
                # token_count += len(encoding.encode(prompt))
        if user_prompt is not None:
            for prompt in to_list(user_prompt):
                total_messages.append(dict(role="user", content=prompt))
                # token_count += len(encoding.encode(prompt))
    else:
        total_messages.extend(messages)

    # 如果不要求用历史对话，那么需要将本轮的对话也记录到历史对话中
    if not using_history and history_message_list is not None:
        history_message_list.extend(total_messages)

    if print_messages:
        print("messages:")
        print_history_message_list(total_messages)
        print()

    # 调用 OpenAI 的 ChatCompletion 接口，不带 tools
    # ChatCompletion(id='chatcmpl-NppKIdfzNiPgzQJRJDSQ1qa3240CP',
    # choices=[Choice(finish_reason='stop', index=0, logprobs=None,
    # message=ChatCompletionMessage(content='我是一个人工智能助手，没有性别和年龄。', role='assistant',
    # function_call=None, tool_calls=None))], created=1706940431, model='gpt-3.5-turbo-0613',
    # object='chat.completion', system_fingerprint=None,
    # usage=CompletionUsage(completion_tokens=20, prompt_tokens=26, total_tokens=46))

    # 带 tools，信息也放在 message 里
    # ChatCompletion(id='chatcmpl-8zTRAg7hokOMb7LGLHj8MPAPiTYC0',
    # choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None,
    # message=ChatCompletionMessage(content=None, role='assistant', function_call=None,
    # tool_calls=[ChatCompletionMessageToolCall(id='call_Z8XsG4KGpIw4TRZ5smJ3cHy1',
    # function=Function(arguments='{\n  "numbers": [2, 4]\n}', name='sum'), type='function')]))],
    # created=1706981485, model='gpt-3.5-turbo-0613',
    # object='chat.completion', system_fingerprint=None,
    # usage=CompletionUsage(completion_tokens=17, prompt_tokens=90, total_tokens=107))
    response = client.chat.completions.create(
        model=model,
        messages=total_messages,
        temperature=temperature,  # 模型输出的温度系数，控制输出的随机程度
        tools=tools
    )

    if print_response:
        print(response)

    choices = response.choices
    choice_count = len(choices)
    # 目前只取第一条
    choice = choices[0]
    message = choice.message
    content = message.content
    finish_reason = choice.finish_reason

    if finish_reason not in {"tool_calls", "stop"}:
        print("IMPORTANT: finish_reason = {finish_reason}.")

    # token_count += len(encoding.encode(content))

    prompt_token_count = response.usage.prompt_tokens
    completion_token_count = response.usage.completion_tokens
    total_token_count = prompt_token_count + completion_token_count
    true_model = response.model
    # from 20240204，gpt-3.5-turbo-0613
    # assert_equal(true_model, "gpt-3.5-turbo-0613")
    # from 20240225，gpt-3.5-turbo-0125
    # assert_equal(true_model, "gpt-3.5-turbo-0125")
    # from 20240304，gpt-35-turbo 或者 gpt-3.5-turbo-0125
    if true_model not in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-16k-0613", "gpt-4-0613"]:
        print(F"IMPORTANT: {true_model}")

    # assert_equal(true_model, "gpt-35-turbo")

    # 无论如何，都保存到历史对话中
    # if not using_history and history_message_list is not None:
    if history_message_list is not None:
        history_message_list.append({"role": "assistant", "content": content})

    end_time = time.time()
    cost_time = end_time - start_time

    if print_token_count:
        print(F"prompt_token_count = {prompt_token_count}, completion_token_count = {completion_token_count}, "
              F"total_token_count = {total_token_count}.")

    if print_cost_time:
        print(F"choice_count = {choice_count}, cost time = {cost_time:.1F}s, finish_reason = {finish_reason}.")
        print()

    # 如果有 tools 参数，那么单独回传 content 已经不够了，需要传回 message
    if tools is not None:
        return message
    else:
        return content


# 将 get_chat_completion_content 更名，方便与下面的 get_chatglm_completion_content 区分
def get_chatgpt_completion_content(user_prompt=None, system_prompt=None, messages=None,
                                   model="gpt-3.5-turbo", temperature=0.1,
                                   print_token_count=False, print_cost_time=False, print_response=False,
                                   history_message_list: List = None,
                                   using_history_message_list=True, tools=None, print_messages=False):
    return get_chat_completion_content(user_prompt=user_prompt, system_prompt=system_prompt, messages=messages,
                                       model=model, temperature=temperature,
                                       print_token_count=print_token_count, print_cost_time=print_cost_time,
                                       print_response=print_response,
                                       history_message_list=history_message_list,
                                       using_history_message_list=using_history_message_list, tools=tools,
                                       print_messages=print_messages)


# 一个封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果
# 图像创建接口只支持一个 prompt
# 推荐用 b64_json + 本地保存的方式做持久化，因为 url 本身就有 1 小时的过期时间
def get_image_create(prompt, model="dall-e-3", response_format="b64_json", size="1024x1024",
                     style="natural", print_response=False, save_image=True, image_name=None, print_prompt=False):
    start_time = time.time()

    client = Client.instance()

    if len(prompt) >= 4000:
        raise ValueError(F"prompt 最多只支持 4000 字的 prompt，目前是 {len(prompt)}.")

    if style not in {"vivid", "natural"}:
        raise ValueError(F"style 只支持 vivid 和 b64_json，目前是 {style}.")

    if response_format not in {"b64_json", "url"}:
        raise ValueError(F"response_format 只支持 b64_json 和 url，目前是 {response_format}.")

    if print_prompt:
        print("prompt:")
        print(prompt)
        print()

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

    # 默认图片名：日期 + 提示词前面 20 个字符 + .png
    local_image_path = (BIGDATA_IMAGE_PATH
                        + (image_name if image_name is not None else create_datetime + "-" + prompt[:20]) + ".png")

    url_or_local_image_path_or_data = url
    if response_format == "url":
        # 如果保存图片的话，返回的就是本地路径，否则返回 url。外面可以统一用 show_image 来展示，已经做了兼容
        if save_image:
            image_data = get_get_response_from_url(url).content
            with open(local_image_path, 'wb') as fp:
                fp.write(image_data)
            url_or_local_image_path_or_data = local_image_path
    else:
        image_data = b64decode(b64_json)
        url_or_local_image_path_or_data = image_data
        # 如果保存图片的话，返回的就是本地路径，否则返回图片数据。外面可以统一用 show_image 来展示，已经做了兼容
        if save_image:
            with open(local_image_path, 'wb') as fp:
                fp.write(image_data)
            url_or_local_image_path_or_data = local_image_path

    end_time = time.time()
    cost_time = end_time - start_time

    print(F"cost time = {cost_time:.1F}s.")

    return revised_prompt, url_or_local_image_path_or_data


# 调试 open 的几个接口 api
# 'check_openai_interfaces' spent 45.4599s.
@func_timer()
def check_openai_interfaces():
    content = get_chat_completion_content(user_prompt=["你是男生女生", "你的年纪是多大？"],
                                          temperature=0.8, print_token_count=True, print_cost_time=True,
                                          print_response=True, print_messages=True)
    print(content)

    revised_prompt, url_or_local_image_path_or_data = get_image_create("展示一只猫和一只狗亲密友好的画面。",
                                                                       response_format="url",
                                                                       print_response=True, print_prompt=True)
    print(revised_prompt, url_or_local_image_path_or_data)

    revised_prompt, url_or_local_image_path_or_data = get_image_create("展示一只猫和一只狗亲密友好的画面。",
                                                                       response_format="b64_json", print_response=True)
    print(revised_prompt, url_or_local_image_path_or_data)


# 利用本地的 whisper 模型进行语音识别，
# model 可能会加载比较慢，如果有多次请求，可以外面加载完毕后传进来即可
def get_whisper_text_local(_path, _model=None, _model_name="base", _language="zh", _initial_prompt=None):
    if _model is None:
        _model = get_whisper_model_local(_model_name)
    return zhconv.convert(_model.transcribe(_path, language=_language, initial_prompt=_initial_prompt)["text"],
                          "zh-cn")


# 从本地获得 whisper 的模型
def get_whisper_model_local(_model_name="base"):
    assert _model_name in {"base", "small", "medium"}
    return whisper.load_model(_model_name, download_root=BIGDATA_WHISPER_PATH)


@func_timer()
def check_whisper_text_local():
    mp3_path = r"D:\PycharmProjects\xiebo\diantou\bigdata\whisper\006018-8746644(0060188746644)_20210714153751.mp3"
    print(get_whisper_text_local(mp3_path))


def main():
    # check_openai_interfaces()
    # check_whisper_text_local()

    prompt = """
    {instruction}

    用户输入：
    {input_text}
    """
    assert_equal(get_format_string_name_list(prompt), ['instruction', 'input_text'])
    print(fill_in_f_string(prompt, instruction="hello"))

    print_closeai()


if __name__ == '__main__':
    main()
