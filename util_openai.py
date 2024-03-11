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


def check_whisper_text_local():
    mp3_path = r"D:\PycharmProjects\xiebo\diantou\bigdata\whisper\006018-8746644(0060188746644)_20210714153751.mp3"
    print(get_whisper_text_local(mp3_path))


def test_f_string():
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
def test_gradio():
    import gradio as gr

    def interact(chatbot_list: List[Tuple[str, str]], user_prompt: str) -> List[Tuple[str, str]]:
        response = get_chatgpt_completion_content(user_prompt=user_prompt)
        chatbot_list.append((user_prompt, response))

        return chatbot_list

    def reset() -> List:
        return list()

    with gr.Blocks() as demo:
        gr.Markdown(F" gradio demo")
        chatbot = gr.Chatbot()
        input_textbox = gr.Textbox(label="input", value="")
        with gr.Row():
            send_button = gr.Button(value="send")
            reset_button = gr.Button(value="reset")

        send_button.click(fn=interact, inputs=[chatbot, input_textbox], outputs=[chatbot])
        reset_button.click(fn=reset, outputs=[chatbot])

    demo.launch(share=True)


# https://www.promptingguide.ai/zh
# 大模型提示工程指南，注意，不同的 chatgpt 版本，效果可能有很大的不同
def test_chatgpt_prompt_engineering():
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
    # test_f_string()
    # test_chatgpt_prompt_engineering()
    test_gradio()

    pass


if __name__ == '__main__':
    main()
