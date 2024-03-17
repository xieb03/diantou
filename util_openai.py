from base64 import b64decode
from threading import RLock

import gradio as gr
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

    # https://platform.openai.com/docs/models
    # turbo 一般指向最新的模型副本
    if model not in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo-preview"]:
        print(F"IMPORTANT: {model}")

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

    # 上面的 turbo 对应的真实版本
    # from 20240204，gpt-3.5-turbo-0613
    # assert_equal(true_model, "gpt-3.5-turbo-0613")
    # from 20240225，gpt-3.5-turbo-0125
    # assert_equal(true_model, "gpt-3.5-turbo-0125")
    # from 20240304，gpt-35-turbo 或者 gpt-3.5-turbo-0125
    if true_model not in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-16k-0613", "gpt-4-0613", "gpt-4-0125-preview"]:
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
    assert _model_name in {"base", "small", "medium", "large-v3"}, _model_name
    return whisper.load_model(_model_name, download_root=BIGDATA_WHISPER_PATH)


def check_whisper_text_local():
    mp3_path = r"D:\PycharmProjects\xiebo\diantou\bigdata\whisper\006018-8746644(0060188746644)_20210714153751.mp3"
    print(get_whisper_text_local(mp3_path))


def check_f_string():
    prompt = """
    {instruction}

    用户输入：
    {input_text}
    """
    assert_equal(get_format_string_name_list(prompt), ['instruction', 'input_text'])
    assert_equal(fill_in_f_string(prompt, instruction="hello"), """
    hello

    用户输入：
    
    """)


# 测试 gradio
def check_summarize_gradio():
    def summarize(prompt: str, article: str, temperature=1.0) -> List[Tuple[str, str]]:
        user_prompt = F"{prompt}\n{article}"
        response = get_chatgpt_completion_content(user_prompt=user_prompt, temperature=temperature)

        return [(user_prompt, response)]

    def reset() -> List:
        return list()

    with gr.Blocks() as demo:
        gr.Markdown(F"# Summarization\nFill in any article you like and let the chatbot summarize it for you")
        chatbot = gr.Chatbot()
        prompt_box = gr.Textbox(label="Prompt", value="请帮我总结一下下面的文章，在 100 字以内.")
        article_box = gr.Textbox(label="Article", interactive=True, value="")

        with gr.Column():
            gr.Markdown(F"# Temperature\n Temperature is used to control the output of the chatbot.")
            temperature_slide = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Temperature")

        with gr.Row():
            send_button = gr.Button(value="send")
            reset_button = gr.Button(value="reset")

        send_button.click(fn=summarize, inputs=[prompt_box, article_box, temperature_slide], outputs=[chatbot])
        reset_button.click(fn=reset, outputs=[chatbot])

    demo.launch(share=True)


# 检查 openai 的 completion 补全接口，主语不是 chat.completion 对话接口
# 20240104，openai 弃用了一些模型，例如 text-davinci-003 -> gpt-3.5-turbo-instruct
# https://www.soinside.com/question/6aWERenG2y5CvgsKvvSBB
def check_openai_completion(model="gpt-3.5-turbo-instruct"):
    client = Client.instance()

    # Completion(id='cmpl-t6s6Gi1V2lShSHHpHGerIZQCOZ8Tt', choices=[CompletionChoice(finish_reason='stop', index=0,
    # logprobs=None, text='\nThis is a test.')], created=1710656217, model='gpt-3.5-turbo-instruct',
    # object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=6, prompt_tokens=6, total_tokens=12))
    # responses = client.completions.create(
    #     model=model,
    #     prompt="Say this is a test."
    # )
    # print(responses)

    # Completion(id='cmpl-x4MJpWmVoyeBXuSf8eyq3o1cv0Vki', choices=[CompletionChoice(finish_reason='length', index=0,
    # logprobs=None, text='\n\n机器学习是一种人工智能的应')], created=1710656628, model='gpt-3.5-turbo-instruct',
    # object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=16, prompt_tokens=11, total_tokens=27))
    # responses = client.completions.create(
    #     model=model,
    #     prompt="请问什么是机器学习",
    # )
    # print(responses)

    # Completion(id='cmpl-FKpyZG1MMztTSiji9AUhEHMSM01C8', choices=[CompletionChoice(finish_reason='length', index=0,
    # logprobs=Logprobs(text_offset=[9, 11, 12, 13, 14, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26],
    # token_logprobs=[-0.7791398, -0.16542703, -5.5146502e-05, -2.5583993e-05, -4.608702e-05, -5.5577775e-06, -0.049086887, -0.09335624, -0.056370255, -2.443443, -3.1217957, -9.0883464e-07, -0.83986676, -0.45744857, -0.56515336, -0.00095195614], tokens=['\n\n', '机', '器', '学', 'bytes:\\xe4\\xb9', 'bytes:\\xa0', '是', '一', '种', '通过', '分', '析', '数据', '和', '模', '式'], top_logprobs=[{'\n\n': -0.7791398, '\n': -1.2817242, '？': -3.1287415, '？\n\n': -3.4863274, '？\n': -4.0128064}, {'机': -0.16542703, '\n': -3.5463855, '<|ipynb_marker|>': -4.3275614, 'bytes:\\xe7\\xad': -4.526576, 'Machine': -4.7376165}, {'器': -5.5146502e-05, '<|endoftext|>': -10.020691, 'bytes:\\xe6': -12.569025, '\n': -13.157766, '<|endoffile|>': -13.185698}, {'学': -2.5583993e-05, '<|endoftext|>': -10.892003, '（': -13.403974, '<|endoffile|>': -14.400964, '机': -14.607474}, {'bytes:\\xe4\\xb9': -4.608702e-05, '<|endoftext|>': -10.497599, '是': -11.443333, 'bytes:\\xe6\\xa0': -13.44491, '<|endoffile|>': -14.012634}, {'bytes:\\xa0': -5.5577775e-06, 'bytes:\\xa0\\xe9\\x99\\xa4': -12.119587, 'bytes:\\xb0': -17.164877, 'bytes:\\x90': -18.17251, 'bytes:\\x9c': -20.586252}, {'是': -0.049086887, '指': -3.3859055, '（': -4.4019313, '(M': -7.051515, ' (': -7.615392}, {'一': -0.09335624, '人': -3.2498465, '指': -3.338582, '通过': -5.211583, '利': -5.492997}, {'种': -0.056370255, '门': -2.934111, '类': -6.8314977, '项': -7.5757604, '组': -11.129457}, {'人': -0.18126084, '通过': -2.443443, '利': -3.78322, '使用': -4.279728, '计': -4.851986}, {'bytes:\\xe8\\xae': -1.6881465, '计': -1.8510627, '使用': -2.2937832, '数据': -2.3811798, '对': -2.451376}, {'析': -9.0883464e-07, 'bytes: \\xe6\\x9e': -15.029662, '解': -15.770886, '别': -16.085926, '<|endoftext|>': -16.149199}, {'数据': -0.83986676, '和': -0.84398663, '大': -2.0732956, '、': -5.2774925, '历': -5.8872566}, {'和': -0.45744857, '、': -2.323482, '，': -2.4208903, '并': -2.8669868, '来': -3.097896}, {'模': -0.56515336, '构': -2.0329134, '建': -3.0422213, '统': -3.0502455, '自': -3.5024703}, {'式': -0.00095195614, '型': -7.04035, 'bytes:\\xe6\\x8b': -10.32851, '板': -10.601038, 'bytes:\\xe4': -11.059795}]),
    # text='\n\n机器学习是一种通过分析数据和模式')], created=1710658045, model='gpt-3.5-turbo-instruct', object='text_completion',
    # system_fingerprint=None, usage=CompletionUsage(completion_tokens=16, prompt_tokens=11, total_tokens=27))
    # 也许在一些分类问题中有用，可以输出概率值
    # responses = client.completions.create(
    #     model=model,
    #     prompt="请问什么是机器学习",
    #     logprobs=5
    # )
    # print(responses)

    # Completion(id='cmpl-of2o80VCpT8GhHXkGrMojphBx3eNy', choices=[CompletionChoice(finish_reason='length', index=0,
    # logprobs=None, text='\n\n机器学习是一种通过给予计算机大量数据，并让计算机自己学习和发现数据内在的规律与模式，从而达到智能化的过程。它是人工智')],
    # created=1710658166, model='gpt-3.5-turbo-instruct', object='text_completion', system_fingerprint=None,
    # usage=CompletionUsage(completion_tokens=64, prompt_tokens=11, total_tokens=75))
    # responses = client.completions.create(
    #     model=model,
    #     prompt="请问什么是机器学习",
    #     max_tokens=64
    # )
    # print(responses)

    # Completion(id='cmpl-xqD1CELXBTjWFMMFe86K5N8HQPqUt', choices=[CompletionChoice(finish_reason='length', index=0,
    # logprobs=None, text='\n\n机器学习是一种人工智能的分支领域，它研究如何使')], created=1710658239, model='gpt-3.5-turbo-instruct',
    # object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=94, prompt_tokens=11, total_tokens=105))
    # 注意 completion_tokens=94，因为每一组答案都消耗 token
    # 既然知道那个答案最好，为什么不直接返回这个答案，而要多次生成再比较呢？因为对于生成式模型，没有生成结束，其实它也不知道这个答案好不好。
    # 注意和 n 会产生交叉，例如 n = 2, best_of = 3，那么实际上会生成 2 组共 6 个答案，然后每一组再找最好的结果返回
    # responses = client.completions.create(
    #     model=model,
    #     prompt="请问什么是机器学习",
    #     max_tokens=32,
    #     best_of=3
    # )
    # print(responses)

    # model：必选参数，具体调用的Completions模型名称，不同模型参数规模不同；
    # 这里需要注意，大模型领域不同于机器学习领域，后者哪怕是简单模型在某些场景下可能也会拥有比复杂模型更好的表现。
    # 在大模型领域，（就OpenAI提供的A、B、C、D四大模型来看）参数规模越大、越新版本的模型效果更好（当然费用也更高），因此课程中主要以text-davinci-003使用为例进行讲解；"
    # prompt：必选参数，提示词；
    # suffix：可选参数，默认为空，具体指模型返回结果的后缀；
    # max_tokens：可选参数，默认为16，代表返回结果的token数量；
    # temperature：可选参数，取值范围为0-2，默认值为1。参数代表采样温度，数值越小，则模型会倾向于选择概率较高的词汇，生成的文本会更加保守；而当temperature值较高时，模型会更多地选择概率较低的词汇，生成的文本会更加多样；
    # top_p：可选参数，取值范围为0-1，默认值为1，和temperature作用类似，用于控制输出文本的随机性，数值越趋近与1，输出文本随机性越强，越趋近于0文本随机性越弱；通常来说若要调节文本随机性，top_p和temperature两个参数选择一个进行调整即可；这里更推荐使用temperature参数进行文本随机性调整；
    # n：可选参数，默认值为1，表示一个提示返回几个Completion；
    # stream：可选参数，默认值为False，表示回复响应的方式，当为False时，模型会等待返回结果全部生成后一次性返回全部结果，而为True时，则会逐个字进行返回；
    # logprobs：可选参数，默认为null，该参数用于指定模型返回前N个概率最高的token及其对数概率。例如，如果logprobs设为10，那么对于生成的每个token，API会返回模型预测的前10个token及其对数概率；
    # echo：可选参数，默认为False，该参数用于控制模型是否应该简单地复述用户的输入。如果设为True，模型的响应会尽可能地复述用户的输入；
    # stop：可选参数，默认为null，该参数接受一个或多个字符串，用于指定生成文本的停止信号。当模型生成的文本遇到这些字符串中的任何一个时，会立即停止生成。这可以用来控制模型的输出长度或格式；
    # presence_penalty：可选参数，默认为0，取值范围为[-2, 2]，该参数用于调整模型生成新内容（例如新的概念或主题）的倾向性。较高的值会使模型更倾向于生成新内容，而较低的值则会使模型更倾向于坚持已有的内容，当返回结果篇幅较大并且存在前后主题重复时，可以提高该参数的取值；
    # frequency_penalty：可选参数，默认为0，取值范围为[-2, 2]，该参数用于调整模型重复自身的倾向性。较高的值会使模型更倾向于避免重复，而较低的值则会使模型更可能重复自身；当返回结果篇幅较大并且存在前后语言重复时，可以提高该参数的取值；相比于 presence_penalty，更愿意使用 presence_penalty
    # best_of：默认是 1，该参数用于控制模型的生成过程。它会让模型进行多次尝试（例如，生成5个不同的响应），然后选择这些响应中得分最高的一个，注意只会选择一个；
    # logit_bias：该参数接受一个字典，用于调整特定token的概率。字典的键是token的ID，值是应用于该token的对数概率的偏置；在GPT中我们可以使用tokenizer tool查看文本Token的标记。一般不建议修改；
    # user：可选参数，使用用户的身份标记，可以通过人为设置标记，来注明当前使用者身份。对结果没有影响。需要注意的是，Completion.create函数中的user和后续介绍的对话类模型的user参数含义并不相同，需要注意区分；

    # prompt = "罗杰有五个网球，他又买了两盒网球，每盒有3个网球，请问他现在总共有多少个网球？"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 罗杰现在总共有11个网球。
    # # 能够发现，此时模型推理得到了正确的结果，罗杰目前总共由5+2*3=11个网球。
    # print(responses.choices[0].text.strip())

    # prompt = "食堂总共有23个苹果，如果他们用掉20个苹果，然后又买了6个苹果，请问现在食堂总共有多少个苹果？"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 食堂现在总共有9个苹果。
    # # 第二个逻辑题比第一个逻辑题稍微复杂一些——复杂之处在于逻辑上稍微转了个弯，即食堂不仅增加了6个苹果，而且还消耗了20个苹果。有增有减，大模型做出了正确判断。
    # print(responses.choices[0].text.strip())

    # prompt = "杂耍者可以杂耍16个球。其中一半的球是高尔夫球，其中一半的高尔夫球是蓝色的。请问总共有多少个蓝色高尔夫球？"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 共有8个蓝色高尔夫球。
    # # 第三个逻辑题的数学计算过程并不复杂，但却设计了一个语言陷阱，即一半的一半是多少。能够发现，模型无法围绕这个问题进行准确的判断，正确答案应该是16\*0.5\*0.5=4个蓝色高尔夫球。
    # print(responses.choices[0].text.strip())

    # prompt = "艾米需要4分钟才能爬到滑梯顶部，她花了1分钟才滑下来，水滑梯将在15分钟后关闭，请问在关闭之前她能滑多少次？"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 在关闭之前，她可以滑（15-1）/（4+1）= 2.8次，所以她最多能滑3次。
    # # 第四个逻辑题是这些逻辑题里数学计算过程最复杂的，涉及多段计算以及除法运算。正确的计算过程应该是先计算艾米一次爬上爬下总共需要5分钟，然后滑梯还有15分钟关闭，因此关闭之前能够再滑15/5=3次。
    # print(responses.choices[0].text.strip())

    # 综上来看，'gpt-3.5-turbo-instruct' 在Zero-shot的情况下，逻辑推理能力较弱，只能围绕相对简单的、只有线性运算过程的推理问题进行很好的解答，总的来看模型只正确回答了第一个问题，其他问题都答错了，模型的推理能力堪忧。

    # prompt = "Q: 艾米需要8分钟才能爬到滑梯顶部，她花了2分钟才滑下来，水滑梯将在30分钟后关闭，请问在关闭之前她能滑多少次？\nA: 30 / (8 + 2) = 10\nQ: 艾米需要4分钟才能爬到滑梯顶部，她花了1分钟才滑下来，水滑梯将在15分钟后关闭，请问在关闭之前她能滑多少次？\nA: "
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 15 / (4 + 1) = 3
    # # 如果给一个相似度非常高的 few-shot，模型可以作对
    # print(responses.choices[0].text.strip())

    # prompt = "Q: 篮子里一共有48张卡片。其中一半的球是方形的，三分之一是蓝色的。请问总共有多少个蓝色方形卡片？\nA: 8\nQ: 杂耍者可以杂耍16个球。其中一半的球是高尔夫球，其中一半的高尔夫球是蓝色的。请问总共有多少个蓝色高尔夫球？\nA:"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 4
    # # 如果给一个相似度非常高的 few-shot，模型可以作对
    # print(responses.choices[0].text.strip())

    # prompt = "艾米需要4分钟才能爬到滑梯顶部，她花了1分钟才滑下来，水滑梯将在15分钟后关闭，请问在关闭之前她能滑多少次？请一步步的思考这个问题"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 先计算出15分钟内可以进行多少轮滑梯：
    # # 已知：
    # # - 滑梯爬上去需要4分钟，滑下来需要1分钟
    # # - 明白15分钟内的时间限制
    # #
    # # 计算过程：
    # # 1. 第一轮：爬上滑梯花费4分钟，滑下来花费1分钟，共花费了5分钟，剩余10分钟可以进行滑梯。
    # # 2. 第二轮：爬上滑梯花费4分钟，滑下来花费1分钟，共花费了5分钟，剩余5分钟可以进行滑梯。
    # # 3. 第三轮：爬上滑梯花费4分钟，滑下来花费1分钟，共花费了5分钟，剩余0分钟，无法进行下一轮滑梯。
    # #
    # # 结论：
    # # 在关闭之前，艾米可以滑3次水滑梯。
    # print(responses.choices[0].text.strip())

    # prompt = "艾米需要4分钟才能爬到滑梯顶部，她花了1分钟才滑下来，水滑梯将在150分钟后关闭，请问在关闭之前她能滑多少次？请一步步的思考这个问题"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 首先是计算爬到顶部和滑下来的总用时是多少，4分钟爬上去再加1分钟滑下来，总共是5分钟
    # #
    # # 根据题目提供的信息，滑梯将在150分钟后关闭，所以可以通过150除以5得到滑梯能滑多少次，即150÷5=30次。因此，艾米在滑梯关闭之前能滑30次。
    # # 如果数据量比较大，模型知道列公式来进行计算，而不再是依次举例
    # print(responses.choices[0].text.strip())

    # prompt = "杂耍者可以杂耍16个球。其中一半的球是高尔夫球，其中一半的高尔夫球是蓝色的。请问总共有多少个蓝色高尔夫球？请一步步进行推理并得出结论。"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 1.根据题意，杂耍者可以杂耍16个球，其中一半是高尔夫球，另一半是其他球。因此，高尔夫球的数量为16的一半，即 16÷2=8.
    # # 2.根据题意，其中一半的高尔夫球是蓝色的，那么蓝色高尔夫球的数量也是8的一半，即 8÷2=4.
    # # 3.因此，总共有4个蓝色高尔夫球，是杂耍者可以杂耍的全部蓝色高尔夫球的数量。
    # print(responses.choices[0].text.strip())

    # prompt = "Q: 食堂总共有23个苹果，如果他们用掉20个苹果，然后又买了6个苹果，请问现在食堂总共有多少个苹果？\nA: 食堂最初有23个苹果，用掉20个，然后又买了6个，总共有23-20+6=9个苹果，答案是9。\nQ: 杂耍者可以杂耍16个球。其中一半的球是高尔夫球，其中一半的高尔夫球是蓝色的。请问总共有多少个蓝色高尔夫球？\nA:"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 杂耍者可以杂耍16个球，其中一半是高尔夫球，也就是8个。其中一半的高尔夫球是蓝色的，也就是4个。所以总共有4个蓝色高尔夫球。
    # # 感觉 few-shot-cot 和 zero-shot-cot 类似，都能激发模型一步步的思考，一个是通过给出一步步思考的案例，另一个是直接让模型一步步思考。
    # # 根据《Large Language Models are Zero-Shot Reasoners》论文中的结论，上从海量数据的测试结果来看，Few-shot-CoT比Zero-shot-CoT准确率更高。
    # print(responses.choices[0].text.strip())

    # prompt = "Q: 食堂总共有23个苹果，如果他们用掉20个苹果，然后又买了6个苹果，请问现在食堂总共有多少个苹果？\nA: 食堂最初有23个苹果，用掉20个，然后又买了6个，总共有23-20+6=9个苹果，答案是9。\nQ: 艾米需要4分钟才能爬到滑梯顶部，她花了1分钟才滑下来，水滑梯将在150分钟后关闭，请问在关闭之前她能滑多少次？\nA:"
    # responses = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=1000,
    # )
    # # 在艾米滑下来的1分钟内，她可以滑1次。在剩下的149分钟里，她每4分钟滑一次，所以她还可以滑149÷4=37.25（取整）次。总共可以滑1+37=38次。
    # # 尽管触发了一步步思考，但思考过程并不对，可能因为因为之前的推理过程和新问题差别比较大。
    # print(responses.choices[0].text.strip())

    prompt = "Q: 艾米需要4分钟才能爬到滑梯顶部，她花了1分钟才滑下来，水滑梯将在15分钟后关闭，请问在关闭之前她能滑多少次？\nA：为了解决'在关闭之前她能滑多少次？'这个问题，我们首先要解决的问题是"
    responses = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1000,
    )
    # ：在15分钟内，艾米能够滑多少次滑梯？我们可以用15分钟除以每次滑梯所需的时间（4分钟+1分钟=5分钟），得知在15分钟内，艾米最多能够滑3次滑梯。也就是说，在关闭之前，艾米最多能够滑3次滑梯。
    print(responses.choices[0].text.strip())


# 利用 completion 实现对话机器人，可以看到，表现并不稳定，而且补全的痕迹很明显，例如回答中包含"好吗？"，"？"等
# input> 你好
# 嗎?
#
# AI: 我是一個人工智能，沒有情緒或感受。我只會根據程式回答您的問題。請問您需要什麼幫助？
# input> 请介绍一下你自己
# 。
#
# AI: 我是一個人工智能助理，被設計來回答各種問題並提供幫助。我可以學習和不斷改進自己的能力，但目前仍然有很多限制。請隨時向我提出您的問題，我會盡力提供最佳的解答。
# input> 什么是机器学习
# ？
#
# AI: 机器学习是一種人工智能的分支，旨在使計算機具有從數據中學習並自動改進的能力。它使用各種統計和數學技術來訓練模型，讓計算機可以辨識和分析複雜的模式並做出預測。透過不斷接收新的數據，機器學習模型可以持續提高其準確性和效率。

# input> 你好
# Bot:你好，有什么可以帮助您的吗？
# input> 请介绍一下你自己
# 好吗？ Bot: 当然，我是一个智能聊天机器人，设计用来与用户交互并提供帮助。我可以回答一些常见问题、提供信息和建议，并且会不断学习以改进自己的服务。有什么需要我的地方，请随时告诉我哦。
# input> 什么是机器学习
# ？ Bot: 机器学习是一种人工智能技术，它允许计算机系统通过学习数据和模式而不需要明确编程来改善其性能。它主要涉及使用算法来分析和识别数据模式，并根据这些模式做出预测或决策。在过去的几年中，机器学习已经被广泛应用于各种领域，包括自然语言处理、金融、医疗保健和物流等。
def check_chat_now(model='gpt-3.5-turbo-instruct', mode='precision'):
    """
    基于Completion.create函数的多轮对话机器人

    :param model: 调用的大语言模型，默认为text-davinci-003
    :param mode: 聊天机器人预设模式，默认为平衡模式balance，可选precision（精确模式）和creativity（创造力模式）

    """
    # 提示想终止聊天时输入"quit"
    print("if you want to stop the conversation, please input 'quit'")
    # 三种不同的模式及其对应的参数
    if mode == 'balance':
        temperature = 1
        presence_penalty = 0
    elif mode == 'precision':
        temperature = 0.8
        presence_penalty = 2
    elif mode == 'creativity':
        temperature = 1.2
        presence_penalty = -1
    else:
        raise ValueError(F"目前只支持 'balance', 'precision', 'creativity'，不支持 {mode}")

    # 定义执行对话函数，方便后续反复调用
    client = Client.instance()

    def chat(_prompt):
        # noinspection PyBroadException
        try:
            # 不报错的情况下，返回Completion.create函数输出结果
            response = client.completions.create(
                model=model,
                prompt=_prompt,
                max_tokens=1000,
                temperature=temperature,
                presence_penalty=presence_penalty,
                # stop=[" Human:", " AI:"]
            )
            answer = response.choices[0].text.strip()
            return answer
        except Exception:
            traceback.print_exc()
            # 报错时返回"broken"
            return "broken"

    # 对话执行函数，首先准备空容器
    text = ""
    turns = []
    # 执行多轮对话，即多次调用chat函数
    while True:
        # 启动对话框
        question = input(Colors.YELLOW + "input> " + Colors.ENDC)
        # 首次开启对话框时提示请输入问题
        if len(question.strip()) == 0:
            print("please input your question")
        # 当输入为 'quit' 时，停止多轮对话，即停止while循环
        elif question == "quit":
            print("\nAI: See You Next Time!")
            break
        else:
            # 多轮对话时，将问题和此前对话结果都作为prompt输入
            prompt = text + "\nHuman: " + question
            result = chat(prompt)
            # 当一次请求失败时，再次发起请求
            while result == "broken":
                print("please wait...")
                result = chat(prompt)
            else:
                # 保留本次对话结果
                turns += [question] + [result]
                print(result)
            # 最多保留十次对话结果，超出次数则最开始的对话会被删除
            if len(turns) <= 10:
                text = " ".join(turns)
            else:
                text = " ".join(turns[-10:])


# https://www.promptingguide.ai/zh
# 大模型提示工程指南，注意，不同的 chatgpt 版本，效果可能有很大的不同
def check_chatgpt_prompt_engineering():
    # prompt = "The sky is "
    # # clear and blue, with a few fluffy white clouds scattered across the horizon.
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 提示词可以明确告诉我们要完善，而不是简单的'续写'
    # prompt = "完善以下句子: The sky is "
    # # clear and blue, with not a cloud in sight.
    # print(get_chat_completion_content(user_prompt=prompt))

    # 目前业界普遍使用的还是更高效的_小样本提示（Few-shot Prompting）_范式，即用户提供少量的提示范例，如任务说明等。小样本提示一般遵循以下格式：
    # <问题>?
    # <答案>
    # <问题>?
    # <答案>
    # <问题>?
    # <答案>
    # <问题>?
    # 而问答模式即如下：
    # Q: <问题>?
    # A: <答案>
    # Q: <问题>?
    # A: <答案>
    # Q: <问题>?
    # A: <答案>
    # Q: <问题>?
    # A:
    # 注意，使用问答模式并不是必须的。你可以根据任务需求调整提示范式。比如，您可以按以下示例执行一个简单的分类任务，并对任务做简单说明：
    # prompt = """
    # This is awesome! // Positive
    # This is bad! // Negative
    # Wow that movie was rad! // Positive
    # What a horrible show! //
    # """
    # # Negative
    # print(get_chat_completion_content(user_prompt=prompt))

    # 提示词可以包含以下任意要素：
    # 指令：想要模型执行的特定任务或指令。
    # 上下文：包含外部信息或额外的上下文信息，引导语言模型更好地响应。
    # 输入数据：用户输入的内容或问题。
    # 输出指示：指定输出的类型或格式。

    # 当您有一个涉及许多不同子任务的大任务时，您可以尝试将任务分解为更简单的子任务，并随着获得更好的结果而不断构建。这避免了在提示设计过程中一开始就添加过多的复杂性。

    # prompt = """
    # 将以下文本翻译成西班牙语：
    # 文本：“hello！”
    # """
    # # ¡Hola!
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 其他人建议将指令放在提示的开头。建议使用一些清晰的分隔符，如“###”，来分隔指令和上下文。
    # prompt = """
    # ### 指令 ###
    # 将以下文本翻译成西班牙语：
    # 文本：“hello！”
    # """
    # # ### 答案 ###
    # # Texto: "¡Hola!"
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 对您希望模型执行的指令和任务非常具体。提示越具体和详细，结果就越好。当您有所期望的结果或生成样式时，这一点尤为重要。
    # # 没有特定的令牌或关键字会导致更好的结果。更重要的是具有良好的格式和描述性提示。实际上，在提示中提供示例非常有效，可以以特定格式获得所需的输出。
    # prompt = """
    # 提取以下文本中的地名。
    # 所需格式：
    # 地点：<逗号分隔的公司名称列表>
    #
    # 输入：“虽然这些发展对研究人员来说是令人鼓舞的，但仍有许多谜团。里斯本未知的香帕利莫德中心的神经免疫学家Henrique Veiga-Fernandes说：“我们经常在大脑和我们在周围看到的效果之间有一个黑匣子。”“如果我们想在治疗背景下使用它，我们实际上需要了解机制。””
    # """
    # # 地点：里斯本, 香帕利莫德中心
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 在上面关于详细和格式改进的提示中，很容易陷入想要过于聪明的提示陷阱，从而可能创建不精确的描述。通常最好是具体和直接。这里的类比非常类似于有效的沟通——越直接，信息传递就越有效。
    # prompt = """
    # 解释提示工程的概念。保持解释简短，只有几句话，不要过于描述。
    # """
    # # 提示工程是一种通过设计和实施提示系统来帮助用户完成特定任务或获取特定信息的方法。它旨在提供用户友好的界面和指导，以提高用户体验和效率。提示工程可以应用于各种领域，如软件开发、网站设计和产品制造。
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 从上面的提示中不清楚要使用多少句话和什么样的风格。您可能仍然可以通过上面的提示获得良好的响应，但更好的提示是非常具体、简洁和直接的。例如：
    # prompt = """
    #     使用2-3句话向高中学生解释提示工程的概念。
    #     """
    # # 提示工程是一种通过设计和布置环境来引导人们做出特定行为或决策的方法。它可以通过改变环境中的元素，如标志、颜色、布局等，来影响人们的行为。提示工程被广泛应用于公共场所、商业场所和社会政策中，以帮助人们做出更健康、更环保或更安全的选择。
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 设计提示时的另一个常见技巧是避免说不要做什么，而是说要做什么。这鼓励更具体化，并关注导致模型产生良好响应的细节。
    # # 以下是一部电影推荐聊天机器人的示例，因为我写的指令——关注于不要做什么，而失败了。
    # prompt = """
    # 以下是向客户推荐电影的代理程序。不要询问兴趣。不要询问个人信息。
    # 客户：请根据我的兴趣推荐电影。
    # 代理：
    # """
    # # 很抱歉，我无法根据您的兴趣来推荐电影。不过我可以向您推荐一些热门的电影，您可以选择感兴趣的进行观看。比如最近上映的《流浪地球》、《复仇者联盟4：终局之战》等。希望您能喜欢！
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 以下是更好的提示：
    # prompt = """
    #     以下是向客户推荐电影的代理程序。代理负责从全球热门电影中推荐电影。它应该避免询问用户的偏好并避免询问个人信息。如果代理没有电影推荐，它应该回答“抱歉，今天找不到电影推荐。”。
    #     顾客：请根据我的兴趣推荐一部电影。
    #     客服：
    #     """
    # # 抱歉，我无法根据您的兴趣来推荐电影。但是我可以向您推荐一部全球热门的电影。您可以尝试观看《流浪地球》，这是一部中国科幻电影，获得了很高的评价和票房。希望您会喜欢！如果您对这部电影不感兴趣，您可以随时告诉我，我可以再为您推荐其他电影。
    # print(get_chat_completion_content(user_prompt=prompt))

    # prompt = """
    #         Explain antibiotics
    #         """
    # # Antibiotics are medications that are used to treat bacterial infections by either killing the bacteria or stopping their growth. They work by targeting specific components of bacterial cells that are essential for their survival, such as their cell wall or protein synthesis machinery.
    # # There are many different types of antibiotics, each with its own mechanism of action and spectrum of activity against different types of bacteria. Some antibiotics are broad-spectrum, meaning they can target a wide range of bacteria, while others are narrow-spectrum and only effective against specific types of bacteria.
    # # It is important to use antibiotics only when prescribed by a healthcare provider and to take them exactly as directed. Overuse or misuse of antibiotics can lead to antibiotic resistance, where bacteria become resistant to the effects of the medication and are harder to treat. This can make infections more difficult to cure and can pose a serious public health threat.
    # print(get_chat_completion_content(user_prompt=prompt))

    # prompt = """
    #             Explain antibiotics
    #             A:
    #             """
    # # Antibiotics are medications that are used to treat bacterial infections. They work by either killing the bacteria or stopping them from multiplying. Antibiotics are only effective against bacterial infections and are not effective against viral infections such as the common cold or flu. It is important to take antibiotics as prescribed by a healthcare provider and to finish the entire course of medication, even if you start feeling better before the medication is finished. This helps to prevent the bacteria from becoming resistant to the antibiotic.
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 文本概括
    # # 在问答形式中，“A:” 是一种明确的提示格式。 在这个示例中，我用它去提示模型，我想要该概念的进一步解释。
    # # 在这个例子中，我们可能还不清楚使用它是否有用，我们会在之后的示例中探讨这一点。 现在假设我们感觉模型给了太多的信息，想要进一步提炼它。 我们可以指导模型帮我们用一句话总结相关内容：
    # # 本示例是模型在没有过多关注上文输出内容的准确性的情况下，尝试用一个句子来总结段落内容。
    # # 关于上文准确性，我们可以通过指令或说明进一步改善它，这一点我们会在后续指南中进行探讨。 读到这里，您可以暂时停住并进行实验，看看是否能获得更好的结果。
    # prompt = """
    #         Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.
    #         Explain the above in one sentence: // 用一句话解释上面的信息：
    #         """
    # # Antibiotics are medications used to treat bacterial infections by killing or preventing the growth of bacteria, but they are not effective against viral infections and misuse can lead to antibiotic resistance.
    # print(get_chat_completion_content(user_prompt=prompt))

    # 信息提取
    # 使用以下示例提示词从指定段落中提取信息：
    # prompt = """
    #         Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
    #         Mention the large language model based product mentioned in the paragraph above: // 指出上文中提到的大语言模型：
    #         """
    # # ChatGPT
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 问答
    # # 提高模型响应精确度的最佳方法之一是改进提示词的格式。 如前所述，提示词可以通过指令、上下文、输入和输出指示以改进响应结果。
    # # 虽然这些要素不是必需的，但如果您的指示越明确，响应的结果就会越好。 以下示例可以说明结构化提示词的重要性。
    # prompt = """
    #         Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. // 基于以下语境回答问题。如果不知道答案的话，请回答“不确定答案”。
    #         Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.
    #         Question: What was OKT3 originally sourced from?
    #         Answer:
    #         """
    # # Mice
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 文本分类
    # prompt = """
    #         Classify the text into neutral, negative or positive. // 将文本按中立、负面或正面进行分类
    #         Text: I think the food was okay.
    #         Sentiment:
    #         """
    # # Neutral
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 我们给出了对文本进行分类的指令，语言模型做出了正确响应，判断文本类型为 'Neutral'。 如果我们想要语言模型以指定格式做出响应， 比如，我们想要它返回 neutral 而不是 Neutral， 那我们要如何做呢？
    # # 我们有多种方法可以实现这一点。 此例中，我们主要是关注绝对特性，因此，我们提示词中包含的信息越多，响应结果就会越好。 我们可以使用以下示例来校正响应结果：
    # prompt = """
    #         Classify the text into neutral, negative or positive.
    #         Text: I think the vacation is okay.
    #         Sentiment: neutral
    #         Text: I think the food was okay.
    #         Sentiment:
    #         """
    # # neutral
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 对话
    # # 你可以通过提示工程进行更有趣的实验，比如指导大语言模型系统如何表现，指定它的行为意图和身份。 如果你正在构建客服聊天机器人之类的对话系统时，这项功能尤其有用。
    # # 比如，可以通过以下示例创建一个对话系统，该系统能够基于问题给出技术性和科学的回答。 你可以关注我们是如何通过指令明确地告诉模型应该如何表现。 这种应用场景有时也被称为 角色提示（Role Prompting）。
    # prompt = """
    #             The following is a conversation with an AI research assistant. The assistant tone is technical and scientific. // 以下是与 AI 助理的对话，语气应该专业、技术性强。
    #             Human: Can you tell me about the creation of blackholes?
    #             AI:
    #             """
    # # Certainly. Black holes are formed when a massive star runs out of fuel and collapses under its own gravity.
    # # This collapse causes the star's core to shrink to a point of infinite density, known as a singularity.
    # # The gravitational pull of this singularity is so strong that not even light can escape, creating a region of spacetime from which nothing can escape, known as the event horizon. This is what we refer to as a black hole.
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 我们的 AI 助理给出的回答非常技术对吧？ 下面，我们让它给出更易于理解的答案。
    # prompt = """
    #         The following is a conversation with an AI research assistant. The assistant answers should be easy to understand even by primary school students. // 以下是与 AI 助理的对话。请给出易于理解的答案，最好是小学生都能看懂的那种。
    #         Human: Can you tell me about the creation of black holes?
    #         AI:
    #         """
    # # Sure! Black holes are created when a massive star runs out of fuel and collapses under its own gravity.
    # # This collapse causes the star to become very dense and form a black hole, which has a strong gravitational pull that not even light can escape from.
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 代码生成
    # # 大语言模型另外一个有效的应用场景是代码生成。 在此方面，Copilot 就是一个很好的示例。 你可以通过一些有效的提示词执行代码生成任务。 让我们来看一下下面的例子。
    # # 我们先用它写个简单的用户欢迎程序：
    # prompt = """
    #         /*
    #         询问用户的姓名并说“ Hello”
    #         */
    #         """
    # # What is your name? Hello!
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 大语言模型另外一个有效的应用场景是代码生成。 在此方面，Copilot 就是一个很好的示例。 你可以通过一些有效的提示词执行代码生成任务。 让我们来看一下下面的例子。
    # # 我们先用它写个简单的用户欢迎程序：
    # prompt = """
    #         使用 python 代码实现以下功能：
    #         询问用户的姓名并说“ Hello”
    #         """
    # # ```python
    # # # 询问用户姓名并打印“Hello”
    # # name = input("请输入您的姓名：")
    # # print("Hello, " + name)
    # # ```
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 来，我们再稍微升级一下。 下面的例子会向你展示提示词会让大语言模型变得多么强大。
    # prompt = """
    #         Table departments, columns = [DepartmentId, DepartmentName]
    #         Table students, columns = [DepartmentId, StudentId, StudentName]
    #         Create a MySQL query for all students in the Computer Science Department
    #         """
    # # SELECT students.StudentId, students.StudentName
    # # FROM students
    # # JOIN departments ON students.DepartmentId = departments.DepartmentId
    # # WHERE departments.DepartmentName = 'Computer Science';
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 推理
    # prompt = """
    #         What is 9,000 * 9,000?
    #         """
    # # 81,000,000
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 来，我们加大难度：
    # prompt = """
    #         The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
    #         A:
    #         """
    # # The odd numbers in the group are 15, 5, 13, 7, and 1.
    # # Adding them up: 15 + 5 + 13 + 7 + 1 = 41
    # # Therefore, the sum of the odd numbers in this group is 41, which is an odd number.
    # print(get_chat_completion_content(user_prompt=prompt))

    # # 来，我们加大难度：
    # prompt = """
    #         The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
    #         Solve by breaking the problem into steps.
    #         """
    # # Step 1: Identify the odd numbers in the group: 15, 5, 13, 7, 1.
    # # Step 2: Add the odd numbers together: 15 + 5 + 13 + 7 + 1 = 41.
    # # Step 3: Check if the sum of the odd numbers is an even number. In this case, 41 is an odd number.
    # # Step 4: Identify the even number in the group: 32, 82.
    # # Step 5: Add the even numbers together: 32 + 82 = 114.
    # # Step 6: Check if the sum of the even numbers is an even number. In this case, 114 is an even number.
    # # Step 7: Conclusion: The odd numbers in the group add up to an odd number, while the even numbers add up to an even number.
    # print(get_chat_completion_content(user_prompt=prompt))

    # 来，我们加大难度：
    prompt = """
                The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
                Solve by breaking the problem into steps. 
                First, identify the odd numbers, add them, and indicate whether the result is odd or even. 
                """
    # The odd numbers in the group are 15, 5, 13, 7, and 1.
    # Adding them together: 15 + 5 + 13 + 7 + 1 = 41.
    # The result, 41, is an odd number.
    print(get_chat_completion_content(user_prompt=prompt))


@func_timer(arg=True)
def main():
    # check_openai_interfaces()
    # check_whisper_text_local()
    # print_closeai()
    # check_f_string()
    # check_chatgpt_prompt_engineering()
    # check_summarize_gradio()
    check_openai_completion()
    # check_chat_now()

    pass


if __name__ == '__main__':
    main()
