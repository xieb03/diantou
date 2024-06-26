# noinspection PyUnresolvedReferences
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 安装 langchain 后要额外安装 pip install pymupdf rapidocr-onnxruntime
from langchain_community.document_loaders import PyMuPDFLoader
# 要额外安装 pip install unstructured
# Resource punkt not found.
# Please use the NLTK Downloader to obtain the resource: nltk.download('punkt')，需要魔术方法，下载到 C:\Users\admin\AppData\Roaming\nltk_data 中
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.documents.base import Document
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from util_chatglm import *
from util_openai import *


# 自定义 langchain + chatglm4
class CHATGLM4LLM(LLM):
    _single_lock = RLock()
    # 温度系数
    temperature: float = 0.2

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              temperature: float = None,
              **kwargs: Any):
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # 在执行的时候也可以重新传入 temperature
        if temperature is None:
            temperature = self.temperature

        # 懒加载，调用 call 才会真正加载模型
        model, tokenizer = CHATGLM4LLM.instance()
        return get_chatglm_completion_content(tokenizer, model, user_prompt=prompt,
                                              temperature=temperature)

    # 单例，返回 model 和 tokenizer
    @classmethod
    def instance(cls):
        # 为了在多线程环境下保证数据安全，在需要并发枷锁的地方加上 RLock 锁
        with cls._single_lock:
            if not hasattr(cls, "_model"):
                cls._model, cls._tokenizer = load_chatglm_model_and_tokenizer(
                    _pretrained_path=GLM4_9B_CHAT_model_dir)
        return cls._model, cls._tokenizer

    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "temperature": self.temperature,
            "model_name": "CustomChatglm4LLM",
        }
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "glm4"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        # return {**{"model": self.model, "tokenizer": self.tokenizer}, **self._default_params}
        return {**self._default_params}


# 自定义 langchain + qwen2
class QWEN2LLM(LLM):
    _single_lock = RLock()
    # 温度系数
    temperature: float = 0.2

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              temperature: float = None,
              **kwargs: Any):
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # 在执行的时候也可以重新传入 temperature
        if temperature is None:
            temperature = self.temperature

        # 懒加载，调用 call 才会真正加载模型
        model, tokenizer = QWEN2LLM.instance()
        return get_qwen2_completion_content(tokenizer, model, user_prompt=prompt,
                                            temperature=temperature)

    # 单例，返回 model 和 tokenizer
    @classmethod
    def instance(cls):
        # 为了在多线程环境下保证数据安全，在需要并发枷锁的地方加上 RLock 锁
        with cls._single_lock:
            if not hasattr(cls, "_model"):
                cls._model, cls._tokenizer = load_chatglm_model_and_tokenizer(
                    _pretrained_path=QWEN2_7B_INSTRUCT_model_dir)
        return cls._model, cls._tokenizer

    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "temperature": self.temperature,
            "model_name": "CustomQwen2LLM",
        }
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "qwen2"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        # return {**{"model": self.model, "tokenizer": self.tokenizer}, **self._default_params}
        return {**self._default_params}


# 获取 langchain 自带的 openAI
def get_langchain_openai_llm(real=False, model="gpt-3.5-turbo", temperature=0.2, max_tokens=None,
                             timeout=None, max_retries=2, llm: ChatOpenAI = None, reset_llm=False):
    if llm is None:
        if real:
            print(F"IMPORTANT 1: real OpenAI")
            openai_base_url, openai_api_key = get_openai_parameter()
        else:
            openai_base_url, openai_api_key = get_closeai_parameter()

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )

    else:
        # 传入了 llm，但仍然用参数重置，一般用不到
        if reset_llm:
            if model is not None:
                llm.model_name = model
            if temperature is not None:
                llm.temperature = temperature
            if max_retries is not None:
                llm.max_retries = max_retries
            if max_tokens is not None:
                llm.max_tokens = max_tokens
            if timeout is not None:
                llm.request_timeout = timeout

    return llm


# 利用 langchain 自带的 openAI 进行聊天
def get_langchain_openai_chat_completion_content(user_prompt=None, system_prompt=None, model="gpt-3.5-turbo",
                                                 temperature=0.2,
                                                 messages=None, print_token_count=False, print_cost_time=False,
                                                 print_response=False,
                                                 history_message_list: List = None,
                                                 using_history_message_list=True, tools=None, print_messages=False,
                                                 strip=False,
                                                 max_tokens=None, real=False, timeout=None, max_retries=2,
                                                 llm: ChatOpenAI = None, reset_llm=False):
    start_time = time.time()

    if user_prompt is not None or system_prompt is not None:
        assert messages is None, "user_prompt 和 system_prompt 为一组，messages 为另外一组，这两组有且只能有一组不为空，目前前者不为空，但是后者也不为空."
    else:
        assert messages is not None, "user_prompt 和 system_prompt 为一组，messages 为另外一组，这两组有且只能有一组不为空，目前前者为空，后者也为空."

    total_messages = list()
    using_history = using_history_message_list and history_message_list is not None and len(history_message_list) != 0

    # 如果明确要求用历史对话，而且传入了历史对话，那么历史对话也要也要进入 prompt
    # 因为 closeAI 是负载均衡的，每次会随机请求不同的服务器，因此是不具备 openAI 自动保存一段历史对话的能力的，需要自己用历史对话来恢复
    # 注意如果同时有 messages 和 history_message_list，那么 history_message_list 会在前
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
    # 否则上面已经自动记录其中
    if not using_history and history_message_list is not None:
        history_message_list.extend(total_messages)

    if print_messages:
        print("messages:")
        print_history_message_list(total_messages)
        print()

    llm = get_langchain_openai_llm(real=real, model=model, temperature=temperature, max_tokens=max_tokens,
                                   timeout=timeout, max_retries=max_retries, llm=llm, reset_llm=reset_llm)

    # content='你好，我是一个智能助手，专门为您提供各种服务和帮助。我可以回答您的问题、提供信息、帮助您解决问题等等。如果您有任何需要，请随时告诉我，
    # 我会尽力帮助您的。感谢您使用我的服务！如果您有任何问题，欢迎随时问我。'
    # response_metadata={'token_usage': {'completion_tokens': 106, 'prompt_tokens': 20, 'total_tokens': 126},
    # 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}
    # id='run-600f3f8b-8cb5-4615-9033-913321eb48a4-0'
    # usage_metadata={'input_tokens': 20, 'output_tokens': 106, 'total_tokens': 126}
    message = llm.invoke(total_messages)
    if print_response:
        print(message)

    content = message.content
    if strip:
        content = content.strip()

    response_metadata = message.response_metadata
    finish_reason = response_metadata["finish_reason"]
    # 主要是判断是不是被截断，例如 length
    if finish_reason not in {"tool_calls", "stop"}:
        print(F"IMPORTANT 2: finish_reason = {finish_reason}.")

    prompt_token_count = response_metadata["token_usage"]["prompt_tokens"]
    completion_token_count = response_metadata["token_usage"]["completion_tokens"]
    total_token_count = prompt_token_count + completion_token_count
    assert_equal(total_token_count, response_metadata["token_usage"]["total_tokens"])
    if print_token_count:
        print(F"prompt_token_count = {prompt_token_count}, completion_token_count = {completion_token_count}, "
              F"total_token_count = {total_token_count}.")

    # noinspection PyUnresolvedReferences
    true_model = response_metadata["model_name"]

    # 上面的 turbo 对应的真实版本
    # from 20240204，gpt-3.5-turbo-0613
    # assert_equal(true_model, "gpt-3.5-turbo-0613")
    # from 20240225，gpt-3.5-turbo-0125
    # assert_equal(true_model, "gpt-3.5-turbo-0125")
    # from 20240304，gpt-35-turbo 或者 gpt-3.5-turbo-0125
    # 在 langchain 中，model 是在创建模型的时候就传入的，而不是在 chat/invoke 传入的
    if llm.model_name not in OPENAI_TRUE_MODEL_DICT:
        print(F"IMPORTANT 3: {llm.model_name}: {true_model}")
    # 注意，与 openAI 的返回不一样
    # elif true_model != OPENAI_TRUE_MODEL_DICT[llm.model_name]:
    elif true_model != llm.model_name:
        print(F"IMPORTANT 4: {llm.model_name}: {true_model}")

    # 无论如何，都保存到历史对话中
    # if not using_history and history_message_list is not None:
    if history_message_list is not None:
        history_message_list.append({"role": "assistant", "content": content})
        # history_message_list.append(message.to_dict())

    end_time = time.time()
    cost_time = end_time - start_time

    if print_cost_time:
        print(F"choice_count = 1, cost time = {cost_time:.1F}s, finish_reason = {finish_reason}.")
        print()

    # 如果有 tools 参数，那么单独回传 content 已经不够了，需要传回 message
    if tools is not None:
        return message
    else:
        return content


# 返回 document 列表的所有字符数
def get_document_list_length(_document_list: List[Document]) -> int:
    return sum(len(document.page_content) for document in _document_list)


# 返回 document 列表的每一个的字符数
def get_document_list_each_length(_document_list: List[Document]) -> List[int]:
    return [len(document.page_content) for document in _document_list]


class LangchainEmbeddings(Embeddings):
    def __init__(self, embedding_model_path=BGE_LARGE_CN_model_dir, max_length=1024, using_gpu=True):
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=embedding_model_path,
            trust_remote_code=True)
        self.embedding_model = AutoModel.from_pretrained(pretrained_model_name_or_path=embedding_model_path,
                                                         trust_remote_code=True)
        self.using_gpu = using_gpu
        if self.using_gpu:
            self.embedding_model = self.embedding_model.cuda()
        self.max_length = max_length

    def embed_query(self, text: str) -> List[float]:
        encoded_input = self.embedding_tokenizer(text, padding=True, truncation=True, max_length=self.max_length,
                                                 return_tensors='pt')
        if self.using_gpu:
            change_dict_value_to_gpu(encoded_input)

        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            # torch.Size([1, n, 1024])
            # print(model_output[0].shape)

            # Perform pooling. In this case, cls pooling.
            # cls-pooling：直接取 [CLS] 的 embedding
            # mean-pooling：取每个 Token 的平均 embedding
            # max-pooling：对得到的每个 Embedding 取 max
            # # torch.Size([1, 1024])
            # sentence_embeddings = model_output[0][:, 0]
            # torch.Size([1024])，因为要求返回的是一维 list
            sentence_embeddings = model_output[0][0, 0]

        # normalize embeddings
        return torch.nn.functional.normalize(sentence_embeddings, p=2, dim=0).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    # 获得两两相似度
    def check_similarities(self, text: str, other_texts: Union[str, List[str]], method="ip"):

        assert method in ["cosine", "l2", "ip"]
        embedding = np.array(self.embed_query(text))
        other_embedding_list = self.embed_documents(to_list(other_texts))
        other_embedding_list = np.array(other_embedding_list, ndmin=2)

        if method == "ip":
            return np.inner(embedding, other_embedding_list)
        elif method == "l2":
            return np.linalg.norm(embedding - other_embedding_list, axis=1)
        elif method == "cosine":
            return np.inner(embedding, other_embedding_list) / (
                    np.linalg.norm(embedding, ord=2, axis=-1) *
                    np.linalg.norm(other_embedding_list, ord=2, axis=-1))
        else:
            raise ValueError(F"not support {method}")


# 清洗 pdf 的 document，注意是原位修改
def clean_pdf_page_content(_document: Document) -> None:
    # 1. 统一换行符为 \n
    # 2. 删除额外的回车，langchain 会在原本两个符号中间插入了 \n
    # 。\n“...”\n。 -> 。“...”。
    # 3. 删除所有汉字之间的空格
    # 4. 将剩余的多个连续空格都变成一个空格
    # TODO：目前还无法处理下面这种换行和段落的区别
    # 前言
    # “周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读
    # 者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推
    # 导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充
    # 具体的推导细节。”
    # 读到这里，大家可能会疑问为啥前面这段话加了引号，因为这只是我们最初的遐想，后来我们了解到，周
    _document.page_content = replace_multiple_spaces(
        delete_all_spaces_between_chinese(delete_one_extra_break_line(replace_break_line(_document.page_content))))


# 清洗 md 的 document，注意是原位修改
def clean_md_page_content(_document: Document) -> None:
    # 1. md文件每一段中间隔了一个换行符，我们同样可以使用replace方法去除。
    _document.page_content = _document.page_content.replace('\n\n', '\n')


# 获取 pdfs 逐页数据，可以选择是否对图片解析，注意是解析图片中的文字，而不返回图片：extract_from_images_with_rapidocr
# 实际测试下来，增加图片解析会极大的减慢速度，而且效果也不是很好

def get_pdf_document_list(_file_path, _extract_images: bool = False, _clean=True) -> List[Document]:
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdfs 文档路径
    loader = PyMuPDFLoader(file_path=_file_path, extract_images=_extract_images)

    # 调用 PyMuPDFLoader Class 的函数 load 对 pdfs 文件进行加载
    # 返回 list, 每一个元素是 langchain_core.documents.base.Document
    page_list = loader.load()
    if _clean:
        for page in page_list:
            clean_pdf_page_content(page)

    return page_list


# 与 get_pdf_document_list 的区别，明确指定只返回一个 document，利用 "".join() 进行合并，同事对 metadata 也进行合并
# 默认的，目前一个 pdf 会按照页来划分，即返回多个 document，这样会对后面的 split_documents 带来影响，因为 split_documents 每一个 document split_text 的结果再拼凑起来，这样会因为分页带来 split 的碎片化
# 可以参见下面 check_get_pdf_document_list 的结果。这里将 pdf 的多个 document 汇总成一个 document，然后再 split，尝试解决这个问题
def get_pdf_document_all_in_one(_file_path, _extract_images: bool = False, _clean=True) -> Document:
    page_list = get_pdf_document_list(_file_path=_file_path, _extract_images=_extract_images, _clean=False)

    page_content = "".join(map(function_get_attr("page_content"), page_list))

    # 校验 metadata，因为来自于同一个 pdf，因此除了 page（表示当前是第几页）不同以外，其它应该全部相同
    all_metadata = None
    for index, page in enumerate(page_list):
        metadata = page.metadata
        del metadata["page"]
        if index == 0:
            all_metadata = metadata
        else:
            assert_equal(all_metadata, metadata)

    document = Document(page_content=page_content, metadata=all_metadata)

    if _clean:
        clean_pdf_page_content(document)

    return document


# 获取 markdown 逐页数据，single 表示整体会返回一个 Document
def get_md_document_list(_file_path, _mode: str = "single", _clean=True) -> List[Document]:
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdfs 文档路径
    loader = UnstructuredMarkdownLoader(file_path=_file_path, mode=_mode)

    # 返回 list, 每一个元素是 langchain_core.documents.base.Document
    md_list = loader.load()

    if _clean:
        for md in md_list:
            clean_md_page_content(md)

    return md_list


# 与 get_md_document_list 的区别，明确指定只返回一个 document
def get_md_document_all_in_one(_file_path, _clean=True) -> Document:
    return get_md_document_list(_file_path=_file_path, _mode="single", _clean=_clean)[0]


def check_get_pdf_document_list():
    pdf_pages = get_pdf_document_list(BIGDATA_PDF_PATH + "pumpkin_book.pdf")
    assert_equal(len(pdf_pages), 196)

    pdf_page = pdf_pages[16]
    # 公式无法很好的解析
    print(f"该文档的描述性数据：{pdf_page.metadata}",
          f"查看该文档的内容:\n{pdf_page.page_content}",
          sep="\n------\n")

    pdf_page = pdf_pages[1]
    # print 可以增加 sep 分隔符
    print(f"该文档的描述性数据：{pdf_page.metadata}",
          f"查看该文档的内容:\n{pdf_page.page_content}",
          sep="\n------\n")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    split_text_list = text_splitter.split_text(pdf_page.page_content)
    # 分割前，文档一共有 1312 个文字.
    # 分割后，一共有 3 段，每段文字分别为 [497, 498, 314]，总文字数是 1309.
    print(F"分割前，文档一共有 {len(pdf_page.page_content)} 个文字.")
    print(
        F"分割后，一共有 {len(split_text_list)} 段，每段文字分别为 {get_str_list_each_length(split_text_list)}，总文字数是 {get_str_list_length(split_text_list)}.")

    print_list(split_text_list)


def check_get_pdf_document_all_in_one():
    document = get_pdf_document_all_in_one(BIGDATA_PDF_PATH + "pumpkin_book.pdf")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    split_text_list = text_splitter.split_text(document.page_content)
    # 分割前，文档一共有 281944 个文字.
    # 分割后，一共有 642 段，每段文字分别为 [470, 497, 492, 450, 479, 481, 471, 492, 491, 463, 478, 421, 476, 467, 494, 475, 480, 476, 481, 474, 468, 472, 471, 461, 476, 474, 466, 467, 489, 464, 464, 464, 488, 498, 464, 478, 466, 491, 480, 456, 476, 476, 480, 429, 484, 474, 490, 480, 389, 454, 227, 478, 481, 385, 499, 186, 56, 498, 160, 378, 498, 212, 229, 476, 284, 499, 499, 235, 398, 420, 387, 498, 498, 498, 498, 105, 498, 493, 498, 67, 373, 422, 474, 381, 498, 493, 350, 499, 84, 499, 74, 295, 498, 134, 499, 181, 485, 382, 499, 81, 495, 415, 364, 348, 322, 289, 499, 497, 241, 493, 494, 487, 462, 470, 495, 490, 457, 463, 476, 462, 475, 489, 487, 474, 494, 466, 464, 479, 482, 491, 466, 466, 477, 475, 478, 473, 485, 490, 461, 492, 498, 498, 490, 497, 473, 479, 474, 437, 456, 457, 476, 472, 488, 489, 499, 445, 473, 453, 494, 490, 477, 469, 469, 470, 496, 461, 497, 499, 490, 485, 495, 467, 476, 478, 497, 475, 475, 469, 477, 482, 496, 473, 492, 408, 478, 489, 468, 486, 499, 475, 482, 492, 448, 476, 478, 469, 481, 451, 486, 491, 496, 433, 452, 473, 456, 496, 478, 498, 490, 478, 475, 484, 474, 486, 497, 492, 382, 464, 498, 330, 279, 456, 464, 429, 493, 468, 499, 499, 484, 471, 499, 498, 441, 420, 482, 468, 459, 470, 484, 498, 452, 489, 493, 495, 493, 341, 444, 492, 468, 455, 499, 477, 477, 357, 498, 492, 462, 498, 485, 491, 476, 480, 486, 495, 413, 494, 499, 490, 461, 451, 491, 454, 484, 482, 492, 430, 499, 498, 497, 496, 489, 483, 497, 496, 499, 478, 434, 452, 461, 470, 499, 488, 490, 481, 441, 492, 450, 474, 452, 499, 479, 493, 499, 491, 477, 466, 479, 496, 495, 496, 485, 433, 499, 445, 461, 463, 471, 451, 474, 485, 472, 468, 440, 485, 498, 498, 492, 469, 485, 431, 447, 460, 498, 483, 459, 470, 498, 479, 459, 473, 472, 486, 449, 491, 456, 489, 447, 481, 453, 466, 448, 460, 441, 471, 444, 471, 480, 454, 460, 455, 492, 471, 496, 490, 492, 475, 483, 492, 490, 472, 489, 493, 468, 493, 472, 479, 480, 497, 480, 461, 413, 496, 496, 497, 376, 494, 275, 469, 487, 456, 456, 463, 434, 486, 495, 493, 496, 438, 442, 440, 450, 490, 498, 446, 472, 492, 495, 484, 480, 457, 479, 497, 471, 467, 497, 470, 496, 495, 496, 495, 491, 497, 481, 475, 490, 469, 462, 475, 356, 472, 493, 490, 489, 451, 461, 470, 480, 433, 452, 496, 376, 488, 458, 497, 493, 474, 476, 464, 381, 471, 436, 492, 456, 489, 492, 471, 464, 488, 473, 481, 464, 499, 493, 490, 499, 443, 464, 422, 499, 496, 480, 459, 467, 452, 494, 497, 450, 484, 490, 466, 475, 487, 486, 481, 479, 488, 412, 406, 499, 472, 496, 499, 466, 449, 448, 481, 484, 495, 241, 492, 462, 459, 480, 458, 498, 497, 441, 473, 493, 396, 483, 490, 481, 434, 473, 414, 471, 464, 492, 458, 307, 481, 495, 478, 462, 485, 495, 497, 477, 486, 486, 459, 447, 411, 477, 465, 499, 488, 497, 492, 458, 492, 472, 451, 482, 488, 449, 487, 495, 468, 487, 345, 499, 194, 463, 436, 473, 469, 495, 490, 489, 466, 494, 486, 484, 495, 495, 488, 478, 489, 476, 496, 470, 469, 465, 465, 469, 468, 499, 486, 458, 499, 475, 492, 494, 468, 471, 424, 467, 427, 496, 468, 469, 443, 446, 487, 488, 478, 475, 450, 486, 487, 496, 491, 469, 456, 475, 429, 496, 486, 483, 442, 450, 483, 497, 456, 493, 468, 478, 465, 491, 497, 432, 491, 499, 449, 481, 438, 465, 460, 453, 481, 464, 466, 476, 473, 496, 484, 453, 493, 443, 458]，总文字数是 296665.
    print(F"分割前，文档一共有 {len(document.page_content)} 个文字.")
    print(
        F"分割后，一共有 {len(split_text_list)} 段，每段文字分别为 {get_str_list_each_length(split_text_list)}，总文字数是 {get_str_list_length(split_text_list)}.")


def check_get_md_document_list():
    md_page = get_md_document_list(BIGDATA_MD_PATH + "1. 简介 Introduction.md")
    assert_equal(len(md_page), 1)

    md_page = md_page[0]
    # 公式无法很好的解析
    print(f"该文档的描述性数据：{md_page.metadata}",
          f"查看该文档的内容:\n{md_page.page_content}",
          sep="\n------\n")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    split_text_list = text_splitter.split_text(md_page.page_content)
    # 分割前，文档一共有 1782 个文字.
    # 分割后，一共有 5 段，每段文字分别为 [322, 416, 300, 349, 391]，总文字数是 1778.
    print(F"分割前，文档一共有 {len(md_page.page_content)} 个文字.")
    print(
        F"分割后，一共有 {len(split_text_list)} 段，每段文字分别为 {get_str_list_each_length(split_text_list)}，总文字数是 {get_str_list_length(split_text_list)}.")

    print_list(split_text_list)


def check_get_md_document_all_in_one():
    md = get_md_document_all_in_one(BIGDATA_MD_PATH + "1. 简介 Introduction.md")

    # 公式无法很好的解析
    print(f"该文档的描述性数据：{md.metadata}",
          f"查看该文档的内容:\n{md.page_content}",
          sep="\n------\n")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    split_text_list = text_splitter.split_text(md.page_content)
    # 分割前，文档一共有 1782 个文字.
    # 分割后，一共有 5 段，每段文字分别为 [322, 416, 300, 349, 391]，总文字数是 1778.
    print(F"分割前，文档一共有 {len(md.page_content)} 个文字.")
    print(
        F"分割后，一共有 {len(split_text_list)} 段，每段文字分别为 {get_str_list_each_length(split_text_list)}，总文字数是 {get_str_list_length(split_text_list)}.")

    print_list(split_text_list)


def check_langchain_embeddings():
    langchain_embeddings = LangchainEmbeddings(embedding_model_path=BGE_LARGE_CN_model_dir)
    embedding_list = langchain_embeddings.embed_documents(["样例数据-1", "样例数据-2", "错例数据-223"])
    # 1024 [0.001481, 0.01648, -0.028145, 0.025962, 0.012677]
    # 1024 [0.015082, 0.004146, -0.015681, 0.03677, 0.017891]
    # 1024 [0.000608, 0.016712, -0.019185, 0.022082, 0.03458]
    for embedding in embedding_list:
        print(len(embedding), list(map(get_round_6, embedding[:5])))


# 获取各种文件类型的文件内容
# 注意每一个 pdf 的每一页都会是一个文档，而每一个 md 都是一个文档
def get_file_document_list(_folder_path_list: Union[str, List[str]], _clean=True) -> List[Document]:
    file_path_list = list_all_file_list(_folder_path_list)

    document_list = list()
    for file_path in file_path_list:
        file_type = file_path.split('.')[-1].lower()
        if file_type == 'pdf':
            # 注意用 extend 而不是 append，因为 返回的也是 list
            document_list.extend(get_pdf_document_list(file_path, _clean=_clean))
        elif file_type == 'md':
            document_list.extend(get_md_document_list(file_path, _clean=_clean))
        else:
            raise ValueError(F"Unsupported file type: {file_type}")

    return document_list


# 获取各种文件类型的文件内容
# 注意用了 all_in_one，每个文件都只会对应一个 document，即使是 pdf
def get_file_document_list_all_in_one(_folder_path_list: Union[str, List[str]], _clean=True) -> List[Document]:
    file_path_list = list_all_file_list(_folder_path_list)

    document_list = list()
    for file_path in file_path_list:
        file_type = file_path.split('.')[-1].lower()
        if file_type == 'pdf':
            document_list.append(get_pdf_document_all_in_one(file_path, _clean=_clean))
        elif file_type == 'md':
            document_list.append(get_md_document_all_in_one(file_path, _clean=_clean))
        else:
            raise ValueError(F"Unsupported file type: {file_type}")

    return document_list


def check_get_file_document_list():
    document_list = get_file_document_list([BIGDATA_MD_PATH, BIGDATA_PDF_PATH])

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    # List[Document]
    split_document_list = text_splitter.split_documents(document_list)
    print_list(split_document_list)
    # 分割前，一共有 205 段，每段文字分别为 [1782, 13385, 11190, 8920, 9816, 14607, 6176, 9428, 441, 53, 1312, 3702, 3971, 4108, 4047, 4067, 4189, 4125, 4219, 4009, 4180, 2133, 1798, 2013, 1521, 1297, 1572, 1490, 1464, 1800, 1282, 1328, 1658, 1929, 1492, 1222, 1338, 1537, 544, 1536, 996, 1285, 1029, 1572, 1138, 1412, 1608, 1069, 950, 1072, 1089, 1160, 1412, 1572, 1171, 1809, 1208, 1463, 1336, 1682, 1716, 363, 1259, 1249, 543, 738, 1263, 1385, 1281, 1163, 1412, 1356, 1621, 1206, 767, 1251, 1018, 1273, 829, 1165, 1268, 1117, 620, 1146, 1467, 1216, 1667, 1157, 1185, 1312, 908, 1291, 1223, 1304, 1186, 1441, 804, 1008, 1257, 1130, 1713, 901, 1233, 1481, 1815, 1518, 1259, 1861, 1183, 1235, 677, 943, 1642, 1409, 1393, 1369, 1944, 1417, 1322, 1592, 1658, 1821, 1147, 1029, 1054, 1193, 1262, 1330, 1268, 1766, 1633, 1508, 870, 1253, 1680, 1211, 1254, 1604, 1719, 1212, 1466, 1271, 1000, 1231, 852, 1554, 1478, 1681, 1512, 1375, 1159, 1186, 1187, 1236, 666, 1891, 338, 1472, 1740, 1597, 1485, 1252, 1665, 1698, 1063, 1242, 1793, 1113, 1505, 1792, 1516, 1528, 1566, 981, 1255, 931, 1217, 956, 916, 1598, 1421, 978, 1356, 1524, 1577, 199, 1516, 1429, 1369, 1356, 1295, 1632, 1768, 2052, 1304, 1381, 1549, 1290, 408, 1083, 665, 811, 1040, 884, 374]，总文字数是 357883.
    # 分割后，一共有 887 段，每段文字分别为 [322, 416, 300, 349, 391, 449, 498, 465, 478, 491, 467, 420, 491, 486, 487, 440, 480, 392, 487, 463, 449, 459, 487, 470, 480, 490, 451, 491, 493, 496, 460, 480, 485, 476, 480, 409, 477, 496, 481, 466, 491, 439, 356, 495, 414, 479, 490, 474, 498, 495, 494, 473, 474, 471, 141, 209, 383, 421, 267, 451, 356, 411, 415, 497, 491, 373, 386, 481, 463, 456, 495, 491, 452, 495, 495, 466, 458, 444, 470, 493, 491, 497, 496, 495, 416, 493, 497, 487, 493, 419, 487, 471, 482, 494, 490, 497, 497, 468, 464, 477, 495, 481, 433, 193, 405, 496, 347, 404, 445, 477, 484, 482, 464, 495, 466, 483, 490, 495, 493, 486, 461, 453, 491, 497, 487, 488, 441, 493, 393, 447, 460, 411, 406, 480, 400, 350, 489, 459, 490, 490, 433, 494, 479, 499, 445, 494, 491, 454, 488, 498, 167, 423, 464, 462, 460, 487, 493, 475, 496, 434, 493, 496, 488, 428, 472, 462, 468, 452, 440, 490, 497, 426, 441, 52, 497, 498, 314, 454, 479, 481, 471, 492, 491, 463, 427, 425, 476, 467, 494, 475, 480, 476, 481, 236, 424, 471, 485, 480, 447, 480, 476, 481, 416, 417, 470, 477, 479, 486, 476, 476, 479, 320, 495, 480, 456, 476, 476, 480, 429, 484, 330, 421, 487, 482, 452, 319, 478, 481, 385, 499, 186, 53, 3, 498, 160, 378, 498, 212, 229, 476, 284, 499, 499, 235, 339, 455, 387, 498, 498, 498, 498, 105, 498, 493, 498, 67, 53, 320, 422, 474, 381, 498, 493, 350, 499, 84, 499, 74, 53, 242, 498, 134, 499, 181, 485, 382, 499, 81, 495, 415, 418, 295, 322, 289, 499, 497, 241, 92, 484, 446, 487, 462, 57, 459, 449, 490, 457, 246, 496, 487, 471, 125, 493, 487, 382, 476, 464, 466, 256, 492, 466, 471, 108, 455, 463, 459, 151, 482, 473, 485, 456, 465, 492, 415, 487, 480, 436, 370, 496, 485, 342, 499, 435, 435, 455, 108, 482, 493, 471, 103, 480, 473, 313, 471, 453, 459, 493, 462, 483, 209, 489, 99, 495, 487, 499, 139, 488, 494, 57, 493, 476, 378, 498, 466, 114, 486, 428, 477, 313, 499, 499, 170, 496, 408, 478, 57, 432, 468, 486, 292, 463, 490, 153, 496, 448, 99, 452, 490, 220, 500, 473, 138, 470, 476, 261, 483, 496, 444, 496, 496, 478, 226, 499, 498, 233, 479, 484, 474, 483, 471, 492, 302, 499, 498, 330, 276, 460, 464, 426, 497, 468, 499, 316, 497, 492, 456, 340, 362, 495, 468, 343, 436, 461, 441, 488, 88, 500, 286, 484, 493, 319, 459, 465, 498, 439, 468, 421, 497, 477, 224, 416, 499, 457, 83, 486, 463, 478, 479, 482, 481, 175, 479, 489, 308, 498, 295, 500, 488, 319, 480, 455, 159, 489, 498, 351, 471, 356, 490, 489, 266, 468, 499, 392, 496, 485, 230, 498, 125, 463, 483, 294, 498, 465, 498, 134, 488, 497, 247, 475, 452, 496, 259, 480, 483, 264, 497, 499, 243, 380, 365, 499, 111, 464, 472, 483, 469, 388, 437, 499, 331, 491, 476, 356, 497, 449, 290, 494, 498, 466, 53, 448, 354, 483, 498, 98, 479, 493, 336, 471, 448, 241, 470, 478, 497, 327, 495, 446, 497, 487, 322, 493, 496, 484, 57, 492, 467, 487, 365, 473, 434, 498, 203, 495, 441, 359, 460, 427, 496, 495, 65, 464, 455, 274, 499, 485, 286, 499, 201, 497, 483, 496, 490, 472, 176, 485, 441, 480, 500, 493, 471, 496, 466, 454, 492, 480, 464, 480, 97, 487, 142, 494, 275, 53, 488, 456, 463, 460, 469, 461, 246, 475, 497, 497, 247, 455, 473, 490, 465, 498, 494, 191, 479, 487, 139, 478, 454, 181, 500, 470, 270, 468, 446, 395, 478, 493, 356, 499, 496, 353, 373, 473, 451, 473, 100, 410, 468, 468, 377, 483, 490, 478, 100, 487, 427, 494, 495, 323, 498, 499, 485, 301, 432, 442, 369, 491, 464, 326, 478, 476, 464, 246, 495, 484, 473, 325, 489, 491, 306, 469, 492, 457, 92, 488, 499, 281, 440, 449, 201, 492, 465, 314, 485, 396, 432, 458, 475, 300, 477, 477, 143, 491, 449, 451, 483, 392, 472, 477, 483, 156, 481, 481, 445, 492, 412, 333, 493, 472, 252, 461, 495, 278, 492, 378, 375, 488, 176, 468, 496, 467, 456, 337, 458, 491, 494, 92, 445, 473, 493, 393, 487, 490, 481, 233, 471, 488, 474, 122, 468, 492, 357, 396, 481, 495, 291, 467, 497, 486, 271, 496, 498, 84, 497, 473, 366, 477, 485, 496, 440, 500, 491, 145, 480, 444, 437, 207, 477, 494, 499, 394, 482, 417, 476, 137, 482, 390, 499, 194, 53, 460, 436, 473, 269, 475, 496, 100, 499, 498, 338, 499, 467, 499, 446, 350, 442, 479, 53, 493, 469, 494, 474, 364, 352, 473, 487, 458, 493, 479, 57, 462, 499, 441, 496, 494, 468, 157, 468, 451, 437, 311, 198, 465, 383, 497, 193, 485, 479, 457, 93, 482, 475, 447, 490, 487, 456, 495, 469, 353, 452, 452, 492, 310, 495, 497, 451, 391, 481, 454, 496, 496, 193, 489, 488, 364, 499, 484, 478, 500, 451, 470, 211, 477, 481, 404, 407, 485, 478, 166, 451, 212, 477, 371, 467, 487, 158, 495, 423, 373]，总文字数是 371772.
    # 可以看到，因为 pdf 这种多页 document 的性质，导致 split 之后有些细碎，尽管指定了 chunk_size = 500，但产生了很多 300 多字的文档
    print(
        F"分割前，一共有 {len(document_list)} 段，每段文字分别为 {get_document_list_each_length(document_list)}，总文字数是 {get_document_list_length(document_list)}.")
    print(
        F"分割后，一共有 {len(split_document_list)} 段，每段文字分别为 {get_document_list_each_length(split_document_list)}，总文字数是 {get_document_list_length(split_document_list)}.")


def check_get_file_document_list_all_in_one():
    document_list = get_file_document_list_all_in_one([BIGDATA_MD_PATH, BIGDATA_PDF_PATH])

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    # List[Document]
    split_document_list = text_splitter.split_documents(document_list)
    print_list(split_document_list)
    # 分割前，一共有 10 段，每段文字分别为 [1782, 13385, 11190, 8920, 9816, 14607, 6176, 9428, 441, 281944]，总文字数是 357689.
    # 分割后，一共有 815 段，每段文字分别为 [322, 416, 300, 349, 391, 449, 498, 465, 478, 491, 467, 420, 491, 486, 487, 440, 480, 392, 487, 463, 449, 459, 487, 470, 480, 490, 451, 491, 493, 496, 460, 480, 485, 476, 480, 409, 477, 496, 481, 466, 491, 439, 356, 495, 414, 479, 490, 474, 498, 495, 494, 473, 474, 471, 141, 209, 383, 421, 267, 451, 356, 411, 415, 497, 491, 373, 386, 481, 463, 456, 495, 491, 452, 495, 495, 466, 458, 444, 470, 493, 491, 497, 496, 495, 416, 493, 497, 487, 493, 419, 487, 471, 482, 494, 490, 497, 497, 468, 464, 477, 495, 481, 433, 193, 405, 496, 347, 404, 445, 477, 484, 482, 464, 495, 466, 483, 490, 495, 493, 486, 461, 453, 491, 497, 487, 488, 441, 493, 393, 447, 460, 411, 406, 480, 400, 350, 489, 459, 490, 490, 433, 494, 479, 499, 445, 494, 491, 454, 488, 498, 167, 423, 464, 462, 460, 487, 493, 475, 496, 434, 493, 496, 488, 428, 472, 462, 468, 452, 440, 490, 497, 426, 441, 470, 497, 492, 450, 479, 481, 471, 492, 491, 463, 478, 421, 476, 467, 494, 475, 480, 476, 481, 474, 468, 472, 471, 461, 476, 474, 466, 467, 489, 464, 464, 464, 488, 498, 464, 478, 466, 491, 480, 456, 476, 476, 480, 429, 484, 474, 490, 480, 389, 454, 227, 478, 481, 385, 499, 186, 56, 498, 160, 378, 498, 212, 229, 476, 284, 499, 499, 235, 398, 420, 387, 498, 498, 498, 498, 105, 498, 493, 498, 67, 373, 422, 474, 381, 498, 493, 350, 499, 84, 499, 74, 295, 498, 134, 499, 181, 485, 382, 499, 81, 495, 415, 364, 348, 322, 289, 499, 497, 241, 493, 494, 487, 462, 470, 495, 490, 457, 463, 476, 462, 475, 489, 487, 474, 494, 466, 464, 479, 482, 491, 466, 466, 477, 475, 478, 473, 485, 490, 461, 492, 498, 498, 490, 497, 473, 479, 474, 437, 456, 457, 476, 472, 488, 489, 499, 445, 473, 453, 494, 490, 477, 469, 469, 470, 496, 461, 497, 499, 490, 485, 495, 467, 476, 478, 497, 475, 475, 469, 477, 482, 496, 473, 492, 408, 478, 489, 468, 486, 499, 475, 482, 492, 448, 476, 478, 469, 481, 451, 486, 491, 496, 433, 452, 473, 456, 496, 478, 498, 490, 478, 475, 484, 474, 486, 497, 492, 382, 464, 498, 330, 279, 456, 464, 429, 493, 468, 499, 499, 484, 471, 499, 498, 441, 420, 482, 468, 459, 470, 484, 498, 452, 489, 493, 495, 493, 341, 444, 492, 468, 455, 499, 477, 477, 357, 498, 492, 462, 498, 485, 491, 476, 480, 486, 495, 413, 494, 499, 490, 461, 451, 491, 454, 484, 482, 492, 430, 499, 498, 497, 496, 489, 483, 497, 496, 499, 478, 434, 452, 461, 470, 499, 488, 490, 481, 441, 492, 450, 474, 452, 499, 479, 493, 499, 491, 477, 466, 479, 496, 495, 496, 485, 433, 499, 445, 461, 463, 471, 451, 474, 485, 472, 468, 440, 485, 498, 498, 492, 469, 485, 431, 447, 460, 498, 483, 459, 470, 498, 479, 459, 473, 472, 486, 449, 491, 456, 489, 447, 481, 453, 466, 448, 460, 441, 471, 444, 471, 480, 454, 460, 455, 492, 471, 496, 490, 492, 475, 483, 492, 490, 472, 489, 493, 468, 493, 472, 479, 480, 497, 480, 461, 413, 496, 496, 497, 376, 494, 275, 469, 487, 456, 456, 463, 434, 486, 495, 493, 496, 438, 442, 440, 450, 490, 498, 446, 472, 492, 495, 484, 480, 457, 479, 497, 471, 467, 497, 470, 496, 495, 496, 495, 491, 497, 481, 475, 490, 469, 462, 475, 356, 472, 493, 490, 489, 451, 461, 470, 480, 433, 452, 496, 376, 488, 458, 497, 493, 474, 476, 464, 381, 471, 436, 492, 456, 489, 492, 471, 464, 488, 473, 481, 464, 499, 493, 490, 499, 443, 464, 422, 499, 496, 480, 459, 467, 452, 494, 497, 450, 484, 490, 466, 475, 487, 486, 481, 479, 488, 412, 406, 499, 472, 496, 499, 466, 449, 448, 481, 484, 495, 241, 492, 462, 459, 480, 458, 498, 497, 441, 473, 493, 396, 483, 490, 481, 434, 473, 414, 471, 464, 492, 458, 307, 481, 495, 478, 462, 485, 495, 497, 477, 486, 486, 459, 447, 411, 477, 465, 499, 488, 497, 492, 458, 492, 472, 451, 482, 488, 449, 487, 495, 468, 487, 345, 499, 194, 463, 436, 473, 469, 495, 490, 489, 466, 494, 486, 484, 495, 495, 488, 478, 489, 476, 496, 470, 469, 465, 465, 469, 468, 499, 486, 458, 499, 475, 492, 494, 468, 471, 424, 467, 427, 496, 468, 469, 443, 446, 487, 488, 478, 475, 450, 486, 487, 496, 491, 469, 456, 475, 429, 496, 486, 483, 442, 450, 483, 497, 456, 493, 468, 478, 465, 491, 497, 432, 491, 499, 449, 481, 438, 465, 460, 453, 481, 464, 466, 476, 473, 496, 484, 453, 493, 443, 458]，总文字数是 375316.
    # 可以看到，结果比上面的 check_get_file_document_list() 略微改善
    print(
        F"分割前，一共有 {len(document_list)} 段，每段文字分别为 {get_document_list_each_length(document_list)}，总文字数是 {get_document_list_length(document_list)}.")
    print(
        F"分割后，一共有 {len(split_document_list)} 段，每段文字分别为 {get_document_list_each_length(split_document_list)}，总文字数是 {get_document_list_length(split_document_list)}.")


def check_langchain_chroma():
    # 获取当前函数名
    collection_name = inspect.currentframe().f_code.co_name

    document_list = get_file_document_list_all_in_one([BIGDATA_MD_PATH, BIGDATA_PDF_PATH])

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    # List[Document]
    split_document_list = text_splitter.split_documents(document_list)
    # 分割前，一共有 10 段，每段文字分别为 [1782, 13385, 11190, 8920, 9816, 14607, 6176, 9428, 441, 281944]，总文字数是 357689.
    # 分割后，一共有 815 段，每段文字分别为 [322, 416, 300, 349, 391, 449, 498, 465, 478, 491, 467, 420, 491, 486, 487, 440, 480, 392, 487, 463, 449, 459, 487, 470, 480, 490, 451, 491, 493, 496, 460, 480, 485, 476, 480, 409, 477, 496, 481, 466, 491, 439, 356, 495, 414, 479, 490, 474, 498, 495, 494, 473, 474, 471, 141, 209, 383, 421, 267, 451, 356, 411, 415, 497, 491, 373, 386, 481, 463, 456, 495, 491, 452, 495, 495, 466, 458, 444, 470, 493, 491, 497, 496, 495, 416, 493, 497, 487, 493, 419, 487, 471, 482, 494, 490, 497, 497, 468, 464, 477, 495, 481, 433, 193, 405, 496, 347, 404, 445, 477, 484, 482, 464, 495, 466, 483, 490, 495, 493, 486, 461, 453, 491, 497, 487, 488, 441, 493, 393, 447, 460, 411, 406, 480, 400, 350, 489, 459, 490, 490, 433, 494, 479, 499, 445, 494, 491, 454, 488, 498, 167, 423, 464, 462, 460, 487, 493, 475, 496, 434, 493, 496, 488, 428, 472, 462, 468, 452, 440, 490, 497, 426, 441, 470, 497, 492, 450, 479, 481, 471, 492, 491, 463, 478, 421, 476, 467, 494, 475, 480, 476, 481, 474, 468, 472, 471, 461, 476, 474, 466, 467, 489, 464, 464, 464, 488, 498, 464, 478, 466, 491, 480, 456, 476, 476, 480, 429, 484, 474, 490, 480, 389, 454, 227, 478, 481, 385, 499, 186, 56, 498, 160, 378, 498, 212, 229, 476, 284, 499, 499, 235, 398, 420, 387, 498, 498, 498, 498, 105, 498, 493, 498, 67, 373, 422, 474, 381, 498, 493, 350, 499, 84, 499, 74, 295, 498, 134, 499, 181, 485, 382, 499, 81, 495, 415, 364, 348, 322, 289, 499, 497, 241, 493, 494, 487, 462, 470, 495, 490, 457, 463, 476, 462, 475, 489, 487, 474, 494, 466, 464, 479, 482, 491, 466, 466, 477, 475, 478, 473, 485, 490, 461, 492, 498, 498, 490, 497, 473, 479, 474, 437, 456, 457, 476, 472, 488, 489, 499, 445, 473, 453, 494, 490, 477, 469, 469, 470, 496, 461, 497, 499, 490, 485, 495, 467, 476, 478, 497, 475, 475, 469, 477, 482, 496, 473, 492, 408, 478, 489, 468, 486, 499, 475, 482, 492, 448, 476, 478, 469, 481, 451, 486, 491, 496, 433, 452, 473, 456, 496, 478, 498, 490, 478, 475, 484, 474, 486, 497, 492, 382, 464, 498, 330, 279, 456, 464, 429, 493, 468, 499, 499, 484, 471, 499, 498, 441, 420, 482, 468, 459, 470, 484, 498, 452, 489, 493, 495, 493, 341, 444, 492, 468, 455, 499, 477, 477, 357, 498, 492, 462, 498, 485, 491, 476, 480, 486, 495, 413, 494, 499, 490, 461, 451, 491, 454, 484, 482, 492, 430, 499, 498, 497, 496, 489, 483, 497, 496, 499, 478, 434, 452, 461, 470, 499, 488, 490, 481, 441, 492, 450, 474, 452, 499, 479, 493, 499, 491, 477, 466, 479, 496, 495, 496, 485, 433, 499, 445, 461, 463, 471, 451, 474, 485, 472, 468, 440, 485, 498, 498, 492, 469, 485, 431, 447, 460, 498, 483, 459, 470, 498, 479, 459, 473, 472, 486, 449, 491, 456, 489, 447, 481, 453, 466, 448, 460, 441, 471, 444, 471, 480, 454, 460, 455, 492, 471, 496, 490, 492, 475, 483, 492, 490, 472, 489, 493, 468, 493, 472, 479, 480, 497, 480, 461, 413, 496, 496, 497, 376, 494, 275, 469, 487, 456, 456, 463, 434, 486, 495, 493, 496, 438, 442, 440, 450, 490, 498, 446, 472, 492, 495, 484, 480, 457, 479, 497, 471, 467, 497, 470, 496, 495, 496, 495, 491, 497, 481, 475, 490, 469, 462, 475, 356, 472, 493, 490, 489, 451, 461, 470, 480, 433, 452, 496, 376, 488, 458, 497, 493, 474, 476, 464, 381, 471, 436, 492, 456, 489, 492, 471, 464, 488, 473, 481, 464, 499, 493, 490, 499, 443, 464, 422, 499, 496, 480, 459, 467, 452, 494, 497, 450, 484, 490, 466, 475, 487, 486, 481, 479, 488, 412, 406, 499, 472, 496, 499, 466, 449, 448, 481, 484, 495, 241, 492, 462, 459, 480, 458, 498, 497, 441, 473, 493, 396, 483, 490, 481, 434, 473, 414, 471, 464, 492, 458, 307, 481, 495, 478, 462, 485, 495, 497, 477, 486, 486, 459, 447, 411, 477, 465, 499, 488, 497, 492, 458, 492, 472, 451, 482, 488, 449, 487, 495, 468, 487, 345, 499, 194, 463, 436, 473, 469, 495, 490, 489, 466, 494, 486, 484, 495, 495, 488, 478, 489, 476, 496, 470, 469, 465, 465, 469, 468, 499, 486, 458, 499, 475, 492, 494, 468, 471, 424, 467, 427, 496, 468, 469, 443, 446, 487, 488, 478, 475, 450, 486, 487, 496, 491, 469, 456, 475, 429, 496, 486, 483, 442, 450, 483, 497, 456, 493, 468, 478, 465, 491, 497, 432, 491, 499, 449, 481, 438, 465, 460, 453, 481, 464, 466, 476, 473, 496, 484, 453, 493, 443, 458]，总文字数是 375316.
    print(
        F"分割前，一共有 {len(document_list)} 段，每段文字分别为 {get_document_list_each_length(document_list)}，总文字数是 {get_document_list_length(document_list)}.")
    print(
        F"分割后，一共有 {len(split_document_list)} 段，每段文字分别为 {get_document_list_each_length(split_document_list)}，总文字数是 {get_document_list_length(split_document_list)}.")

    langchain_embeddings = LangchainEmbeddings(embedding_model_path=BGE_LARGE_CN_model_dir)

    # 内部调用 get_or_create_collection，因此不存在会新建，否则沿用
    chroma_db = Chroma(embedding_function=langchain_embeddings,
                       persist_directory=CHROMADB_PATH,
                       collection_name=collection_name,
                       # 指定相似度用内积，支持 cosine, l2, ip
                       collection_metadata={"hnsw:space": "ip"})
    assert_equal(len(chroma_db), 0)

    try:
        chroma_db.add_documents(documents=split_document_list[:10])
        # Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
        # chroma_db.persist()

        assert_equal(len(chroma_db), 10)

        query = "什么是大语言模型"
        k = 3

        # where: A Where type dict used to filter results by. E.g. `{"$and": [{"color" : "red"}, {"price": {"$gte": 4.20}}]}`. Optional.
        # 从 metadata 里面筛选
        where = None
        # where_document: A WhereDocument type dict used to filter by the documents. E.g. `{$contains: {"text": "hello"}}`. Optional.
        # 从 document 中筛选
        where_document = None

        # 调用的是 similarity_search_with_score，只不过将 score 去掉了
        # chroma_db.similarity_search(query, k, where_document=where_document)

        # Run similarity search with Chroma with distance，注意返回是距离而不是相似度
        sim_document_list = chroma_db.similarity_search_with_score(query, k, filter=where,
                                                                   where_document=where_document)
        answer_list = [query]
        # 检索到的第0个内容 0.628823: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\1. 简介 Introduction.md', 'start_index': 323}, 网络上有许多关于提示词（Prompt， 本教程中将保留该术语）设计的材料，例如《30 prompts
        # 检索到的第1个内容 0.517759: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\1. 简介 Introduction.md', 'start_index': 1041}, 与基础语言模型不同，指令微调 LLM 通过专门的训练，可以更好地理解并遵循指令。举个例子，当询问“法
        # 检索到的第2个内容 0.510242: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\2. 提示原则 Guidelines.md', 'start_index': 433}, 一、原则一 编写清晰、具体的指令
        # 亲爱的读者，在与语言模型交互时，您需要牢记一点:以清晰、具体的方式
        for i, (sim_doc, distance) in enumerate(sim_document_list):
            answer_list.append(sim_doc.page_content)
            print(f"检索到的第{i}个内容 {1 - distance:.6f}: {sim_doc.metadata}, {sim_doc.page_content[:50]}")

        # [0.99999995 0.62882254 0.51775886 0.51024218]
        print(langchain_embeddings.check_similarities(query, answer_list))
        # [1.         0.62882253 0.51775886 0.5102422 ]
        print(langchain_embeddings.check_similarities(query, answer_list, method="cosine"))

        # 将 score 的范围限定在 0.0 ~ 1.0 之间，但并不一定是准确的，例如用内积的时候，distance 转相关性用的是如下的公式，对于负数的处理感觉不太准确，因为官网给的 d = 1 - ip
        # https://docs.trychroma.com/guides
        #     def _max_inner_product_relevance_score_fn(distance: float) -> float:
        #         """Normalize the distance to a score on a scale [0, 1]."""
        #         if distance > 0:
        #             return 1.0 - distance
        #
        #         return -1.0 * distance
        sim_document_list = chroma_db.similarity_search_with_relevance_scores(query, k, filter=where,
                                                                              where_document=where_document)
        for i, (sim_doc, similarity) in enumerate(sim_document_list):
            print(f"检索到的第{i}个内容 {similarity:.6f}: {sim_doc.metadata}, {sim_doc.page_content[:50]}")

        # 利用 where_document 限制 metadata
        sim_document_list = chroma_db.similarity_search_with_score(query, k, filter=None,
                                                                   # 要求 document 中必须含有如下子串
                                                                   where_document={"$contains": "例子"})
        # 检索到的第0个内容 0.517759: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\1. 简介 Introduction.md', 'start_index': 1041}, 与基础语言模型不同，指令微调 LLM 通过专门的训练，可以更好地理解并遵循指令。举个例子，当询问“法
        # 检索到的第1个内容 0.339630: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\2. 提示原则 Guidelines.md', 'start_index': 885}, 在编写 Prompt 时，我们可以使用各种标点符号作为“分隔符”，将不同的文本部分区分开来。
        for i, (sim_doc, distance) in enumerate(sim_document_list):
            print(f"检索到的第{i}个内容 {1 - distance:.6f}: {sim_doc.metadata}, {sim_doc.page_content[:50]}")

        # 利用 filter 限制 metadata
        # noinspection PyTypeChecker
        sim_document_list = chroma_db.similarity_search_with_score(query, k,
                                                                   filter={"$and": [{"start_index": {"$lte": 2000}},
                                                                                    {"start_index": {"$gte": 1000}}]},
                                                                   # 要求 document 中必须含有如下子串
                                                                   where_document={"$contains": "例子"})
        # 检索到的第0个内容 0.517759: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\1. 简介 Introduction.md', 'start_index': 1041}, 与基础语言模型不同，指令微调 LLM 通过专门的训练，可以更好地理解并遵循指令。举个例子，当询问“法
        for i, (sim_doc, distance) in enumerate(sim_document_list):
            print(f"检索到的第{i}个内容 {1 - distance:.6f}: {sim_doc.metadata}, {sim_doc.page_content[:50]}")

        # 用于根据向量搜索相似的文本，并把结果根据 mmr（max marginal relevance）重新排序，同时考虑相关性和多样性
        # 需要指定预先 fetch 的数量，和 lambda 参数
        sim_document_list = chroma_db.max_marginal_relevance_search(query, k, fetch_k=10, lambda_mult=0.5, filter=where,
                                                                    where_document=where_document)
        # 检索到的第0个内容 0.628823: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\1. 简介 Introduction.md', 'start_index': 323}, 网络上有许多关于提示词（Prompt， 本教程中将保留该术语）设计的材料，例如《30 prompts
        # 检索到的第1个内容 0.517759: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\1. 简介 Introduction.md', 'start_index': 1041}, 与基础语言模型不同，指令微调 LLM 通过专门的训练，可以更好地理解并遵循指令。举个例子，当询问“法
        # 检索到的第2个内容 0.357993: {'source': 'D:\\PycharmProjects\\xiebo\\diantou\\bigdata\\mds\\2. 提示原则 Guidelines.md', 'start_index': 1758}, prompt = f"""
        # 请生成包括书名、作者和类别的三本虚构的、非真实存在的中文书籍清单，\
        # 并
        for i, sim_doc in enumerate(sim_document_list):
            distance = 1 - langchain_embeddings.check_similarities(query, sim_doc.page_content)
            print(f"检索到的第{i}个内容 {1 - distance[0]:.6f}: {sim_doc.metadata}, {sim_doc.page_content[:50]}")
    finally:
        # 完整删除 collection，不光是 length = 0，而是完全不存在
        chroma_db.delete_collection()


def check_get_langchain_openai_chat_completion_content():
    llm = get_langchain_openai_llm(model="gpt-4-turbo", temperature=0.2, real=False)

    system_prompt = "你是一个小学一年级的老师。"
    user_prompt = "请你自我介绍一下自己！"

    # 大家好，我是你们的一年级老师，我在教育领域工作多年，特别喜欢和小朋友们一起学习新事物。我的教学风格温和而富有耐心，我相信每个孩子都有自己独特的学习方式，
    # 我会尽力帮助你们找到最适合自己的学习方法。在课堂上，我们不仅会学习语文、数学等基础知诀，还会有很多有趣的活动来帮助你们更好地理解和运用所学知识。
    # 希望我们能一起度过一个充满乐趣和学习的美好时光！
    content = get_langchain_openai_chat_completion_content(user_prompt=user_prompt, system_prompt=system_prompt,
                                                           llm=llm)
    print(content)


def check_langchain_chat_prompt_template():
    template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    text = "我带着比身体重的行李，游入尼罗河底，经过几道闪电 看到一堆光圈，不确定是不是这里。"
    messages = chat_prompt.format_messages(input_language="中文", output_language="英文", text=text)

    # [SystemMessage(content='你是一个翻译助手，可以帮助我将 中文 翻译成 英文.'),
    # HumanMessage(content='我带着比身体重的行李，游入尼罗河底，经过几道闪电 看到一堆光圈，不确定是不是这里。')]
    print(messages)


# 简单的 langchain icel 实例
def check_langchain_icel():
    template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    llm = get_langchain_openai_llm(model="gpt-4-turbo", temperature=0.2, real=False)

    output_parser = StrOutputParser()

    chain = chat_prompt | llm | output_parser

    text = "我带着比身体重的行李，游入尼罗河底，经过几道闪电 看到一堆光圈，不确定是不是这里。"
    output = chain.invoke({"input_language": "中文", "output_language": "英文", "text": text})
    # I carried luggage heavier than my body weight, diving into the bottom of the Nile River.
    # After passing through several flashes of lightning, I saw a bunch of halos, unsure if this is the place.
    print(output)

    text = 'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'
    output = chain.invoke({"input_language": "英文", "output_language": "中文", "text": text})
    # 我背着比我身体还重的行李，潜入尼罗河的底部。穿过几道闪电之后，我看到了一堆光环，不确定这是不是那个地方。
    print(output)


# 自定义 langchain + chatglm4
def check_custom_chatglm4_llm():
    llm = CHATGLM4LLM()
    # Params: {'temperature': 0.2, 'model_name': 'CustomChatglm4LLM'}
    print(llm)
    # LangChainDeprecationWarning: The method `BaseLLM.__call__` was deprecated in langchain-core 0.1.7 and will be removed in 0.3.0. Use invoke instead.
    # print(llm("老鼠生病了，可以吃老鼠药治好么？"))
    print(llm.invoke("老鼠生病了，可以吃老鼠药治好么？"))
    print(llm.invoke("为什么苏东坡不能参加苏轼的葬礼？", temperature=0.8))
    # invoke 最后还是调用的 __call__ 方法，但注意，即使传入的是一个 list，还是会被认为是一次对话，底层会处理为 "
    # 'Human: 老鼠生病了，可以吃老鼠药治好么？
    # Human: 为什么苏东坡不能参加苏轼的葬礼？'"
    # 即用 \n 将多个元素拼接起来，作为整体的问句。
    # 关于第一个问题，老鼠生病了是否可以吃老鼠药来治疗，这实际上是一个错误的做法。老鼠药通常是用来控制老鼠数量或者用于农业害虫防治的，它的成分对老鼠来说是致命的，对人类同样也是有害的。如果家中的宠物老鼠生病了，应该带它去看兽医，由专业的兽医来诊断和治疗。
    # 至于第二个问题，苏东坡和苏轼实际上是同一个人。苏轼（1037-1101），字子瞻，号东坡居士，是中国北宋时期的文学家、书画家、政治家，而苏东坡是他的号。因此，苏东坡就是苏轼，他当然可以参加自己的葬礼。这里可能是一个误解或者是一个玩笑。在现实生活中，人无法参加自己的葬礼，因为葬礼是在人去世后举行的。
    print(llm.invoke(["老鼠生病了，可以吃老鼠药治好么？", "为什么苏东坡不能参加苏轼的葬礼？"], temperature=0.8))


def check_a_whole_rag_demo():
    # 获取当前函数名
    collection_name = inspect.currentframe().f_code.co_name

    document_list = get_file_document_list_all_in_one([BIGDATA_MD_PATH, BIGDATA_PDF_PATH])

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    # List[Document]
    split_document_list = text_splitter.split_documents(document_list)
    # 分割前，一共有 10 段，每段文字分别为 [1782, 13385, 11190, 8920, 9816, 14607, 6176, 9428, 441, 281944]，总文字数是 357689.
    # 分割后，一共有 815 段，每段文字分别为 [322, 416, 300, 349, 391, 449, 498, 465, 478, 491, 467, 420, 491, 486, 487, 440, 480, 392, 487, 463, 449, 459, 487, 470, 480, 490, 451, 491, 493, 496, 460, 480, 485, 476, 480, 409, 477, 496, 481, 466, 491, 439, 356, 495, 414, 479, 490, 474, 498, 495, 494, 473, 474, 471, 141, 209, 383, 421, 267, 451, 356, 411, 415, 497, 491, 373, 386, 481, 463, 456, 495, 491, 452, 495, 495, 466, 458, 444, 470, 493, 491, 497, 496, 495, 416, 493, 497, 487, 493, 419, 487, 471, 482, 494, 490, 497, 497, 468, 464, 477, 495, 481, 433, 193, 405, 496, 347, 404, 445, 477, 484, 482, 464, 495, 466, 483, 490, 495, 493, 486, 461, 453, 491, 497, 487, 488, 441, 493, 393, 447, 460, 411, 406, 480, 400, 350, 489, 459, 490, 490, 433, 494, 479, 499, 445, 494, 491, 454, 488, 498, 167, 423, 464, 462, 460, 487, 493, 475, 496, 434, 493, 496, 488, 428, 472, 462, 468, 452, 440, 490, 497, 426, 441, 470, 497, 492, 450, 479, 481, 471, 492, 491, 463, 478, 421, 476, 467, 494, 475, 480, 476, 481, 474, 468, 472, 471, 461, 476, 474, 466, 467, 489, 464, 464, 464, 488, 498, 464, 478, 466, 491, 480, 456, 476, 476, 480, 429, 484, 474, 490, 480, 389, 454, 227, 478, 481, 385, 499, 186, 56, 498, 160, 378, 498, 212, 229, 476, 284, 499, 499, 235, 398, 420, 387, 498, 498, 498, 498, 105, 498, 493, 498, 67, 373, 422, 474, 381, 498, 493, 350, 499, 84, 499, 74, 295, 498, 134, 499, 181, 485, 382, 499, 81, 495, 415, 364, 348, 322, 289, 499, 497, 241, 493, 494, 487, 462, 470, 495, 490, 457, 463, 476, 462, 475, 489, 487, 474, 494, 466, 464, 479, 482, 491, 466, 466, 477, 475, 478, 473, 485, 490, 461, 492, 498, 498, 490, 497, 473, 479, 474, 437, 456, 457, 476, 472, 488, 489, 499, 445, 473, 453, 494, 490, 477, 469, 469, 470, 496, 461, 497, 499, 490, 485, 495, 467, 476, 478, 497, 475, 475, 469, 477, 482, 496, 473, 492, 408, 478, 489, 468, 486, 499, 475, 482, 492, 448, 476, 478, 469, 481, 451, 486, 491, 496, 433, 452, 473, 456, 496, 478, 498, 490, 478, 475, 484, 474, 486, 497, 492, 382, 464, 498, 330, 279, 456, 464, 429, 493, 468, 499, 499, 484, 471, 499, 498, 441, 420, 482, 468, 459, 470, 484, 498, 452, 489, 493, 495, 493, 341, 444, 492, 468, 455, 499, 477, 477, 357, 498, 492, 462, 498, 485, 491, 476, 480, 486, 495, 413, 494, 499, 490, 461, 451, 491, 454, 484, 482, 492, 430, 499, 498, 497, 496, 489, 483, 497, 496, 499, 478, 434, 452, 461, 470, 499, 488, 490, 481, 441, 492, 450, 474, 452, 499, 479, 493, 499, 491, 477, 466, 479, 496, 495, 496, 485, 433, 499, 445, 461, 463, 471, 451, 474, 485, 472, 468, 440, 485, 498, 498, 492, 469, 485, 431, 447, 460, 498, 483, 459, 470, 498, 479, 459, 473, 472, 486, 449, 491, 456, 489, 447, 481, 453, 466, 448, 460, 441, 471, 444, 471, 480, 454, 460, 455, 492, 471, 496, 490, 492, 475, 483, 492, 490, 472, 489, 493, 468, 493, 472, 479, 480, 497, 480, 461, 413, 496, 496, 497, 376, 494, 275, 469, 487, 456, 456, 463, 434, 486, 495, 493, 496, 438, 442, 440, 450, 490, 498, 446, 472, 492, 495, 484, 480, 457, 479, 497, 471, 467, 497, 470, 496, 495, 496, 495, 491, 497, 481, 475, 490, 469, 462, 475, 356, 472, 493, 490, 489, 451, 461, 470, 480, 433, 452, 496, 376, 488, 458, 497, 493, 474, 476, 464, 381, 471, 436, 492, 456, 489, 492, 471, 464, 488, 473, 481, 464, 499, 493, 490, 499, 443, 464, 422, 499, 496, 480, 459, 467, 452, 494, 497, 450, 484, 490, 466, 475, 487, 486, 481, 479, 488, 412, 406, 499, 472, 496, 499, 466, 449, 448, 481, 484, 495, 241, 492, 462, 459, 480, 458, 498, 497, 441, 473, 493, 396, 483, 490, 481, 434, 473, 414, 471, 464, 492, 458, 307, 481, 495, 478, 462, 485, 495, 497, 477, 486, 486, 459, 447, 411, 477, 465, 499, 488, 497, 492, 458, 492, 472, 451, 482, 488, 449, 487, 495, 468, 487, 345, 499, 194, 463, 436, 473, 469, 495, 490, 489, 466, 494, 486, 484, 495, 495, 488, 478, 489, 476, 496, 470, 469, 465, 465, 469, 468, 499, 486, 458, 499, 475, 492, 494, 468, 471, 424, 467, 427, 496, 468, 469, 443, 446, 487, 488, 478, 475, 450, 486, 487, 496, 491, 469, 456, 475, 429, 496, 486, 483, 442, 450, 483, 497, 456, 493, 468, 478, 465, 491, 497, 432, 491, 499, 449, 481, 438, 465, 460, 453, 481, 464, 466, 476, 473, 496, 484, 453, 493, 443, 458]，总文字数是 375316.
    print(
        F"分割前，一共有 {len(document_list)} 段，每段文字分别为 {get_document_list_each_length(document_list)}，总文字数是 {get_document_list_length(document_list)}.")
    print(
        F"分割后，一共有 {len(split_document_list)} 段，每段文字分别为 {get_document_list_each_length(split_document_list)}，总文字数是 {get_document_list_length(split_document_list)}.")

    langchain_embeddings = LangchainEmbeddings(embedding_model_path=BGE_LARGE_CN_model_dir)

    # 内部调用 get_or_create_collection，因此不存在会新建，否则沿用
    chroma_db = Chroma(embedding_function=langchain_embeddings,
                       persist_directory=CHROMADB_PATH,
                       collection_name=collection_name,
                       # 指定相似度用内积，支持 cosine, l2, ip
                       collection_metadata={"hnsw:space": "ip"})
    assert_equal(len(chroma_db), 0)

    try:
        chroma_db.add_documents(documents=split_document_list)
        assert_equal(len(chroma_db), 815)

        # llm = CHATGLM4LLM()
        # llm = QWEN2LLM()
        llm = get_langchain_openai_llm(model="gpt-4-turbo", temperature=0.2, real=False)

        # template = """抛开以前的知识，只能根据以下上下文来回答最后的问题，如果无法根据上下文来回答，就说你不知道，不要试图编造答
        # 案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        # {context}
        # 问题: {question}
        # """

        # template = """从文档 “““{context}””” 中找问题 “““{question}””” 的答案，能找到答案就借助文档回答问题，否则就说不知道。
        # 不要试图编造答案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        # """

        template = """
        从文档
        \"\"\"
        {context}
        \"\"\"
        中找问题
        \"\"\"
        {question}
        \"\"\"
        的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。
        不要复述问题，直接开始回答。
        """
        # import langchain
        # langchain.debug = True

        chain_prompt = PromptTemplate(template=template)

        # llm：指定使用的 LLM
        # 指定 chain type : RetrievalQA.from_chain_type(chain_type="stuff")，也可以利用load_qa_chain()方法指定chain type。
        # 自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}
        # 返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）
        qa_chain = RetrievalQA.from_chain_type(llm,
                                               chain_type="stuff",
                                               retriever=chroma_db.as_retriever(search_type="similarity",
                                                                                search_kwargs={'k': 3}),
                                               return_source_documents=False,
                                               chain_type_kwargs={"prompt": chain_prompt})

        question_1 = "什么是南瓜书？"
        question_2 = "王阳明是谁？"

        result = qa_chain({"query": question_1})
        print("大模型 + 知识库后回答 question_1 的结果：")
        # 我不知道什么是南瓜书，因为提供的上下文中没有相关信息。
        print(result["result"], end="\n-----------------------\n")

        result = qa_chain({"query": question_2})
        print("大模型 + 知识库后回答 question_2 的结果：")
        # 王阳明是中国明代著名的哲学家、思想家、军事家和教育家，也是心学的集大成者。他提出了“致良知”的理念，强调内心修养的重要性。王阳明的学说对后世产生了深远影响。
        print(result["result"], end="\n-----------------------\n")

        print("大模型独立回答 question_1 的结果：")
        result = llm.invoke(question_1)
        # “南瓜书”这个说法可能源自于网络上的一个梗或者特定的语境，但通常并没有一个明确、广泛接受的定义。如果“南瓜书”指的是某个特定的书籍或系列书籍，那么可能需要提供更多的上下文信息来准确回答。在一些文化或网络社区中，“南瓜书”可能被用来描述某种类型的书籍，比如可能是以南瓜为主题、与万圣节相关的读物，或者是某种风格独特、内容特别的书籍。然而，没有足够的信息来确定“南瓜书”的确切含义，因此在回答时需要保持一定的灵活性和开放性。如果你能提供更多关于“南瓜书”的具体信息，我或许能给出更精确的回答。
        print(result if isinstance(result, str) else result.content, end="\n-----------------------\n")

        print("大模型独立回答 question_2 的结果：")
        result = llm.invoke(question_2)
        # 王阳明，名守仁，字伯安，号阳明子，故被称为王阳明。他是明代著名的思想家、哲学家、教育家、军事家及文学家，也是心学的集大成者。王阳明的学术思想主要体现在“心即理”、“知行合一”和“致良知”三个方面。
        # 王阳明在明朝中叶提出的心学理论，对后世产生了深远的影响，不仅在中国文化界有着重要的地位，在东亚乃至全球范围内也具有广泛的文化影响力。他的思想强调内心修养与道德实践的重要性，认为人的心灵本自具足，通过自我反省和实践可以达到道德完善和个人成长的目的。
        # 此外，王阳明还是一位杰出的军事战略家，在平定宁王朱宸濠叛乱中表现出卓越的军事才能，为明朝的稳定做出了贡献。他的文笔流畅，作品涵盖了诗歌、散文、小说等多个领域，留下了许多脍炙人口的作品。
        print(result if isinstance(result, str) else result.content, end="\n-----------------------\n")

    finally:
        # 完整删除 collection，不光是 length = 0，而是完全不存在
        chroma_db.delete_collection()


def check_a_whole_rag_demo_with_memory():
    # 获取当前函数名
    collection_name = inspect.currentframe().f_code.co_name

    document_list = get_file_document_list_all_in_one([BIGDATA_MD_PATH, BIGDATA_PDF_PATH])

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, chunk_overlap=50, add_start_index=True)

    # List[Document]
    split_document_list = text_splitter.split_documents(document_list)
    # 分割前，一共有 10 段，每段文字分别为 [1782, 13385, 11190, 8920, 9816, 14607, 6176, 9428, 441, 281944]，总文字数是 357689.
    # 分割后，一共有 815 段，每段文字分别为 [322, 416, 300, 349, 391, 449, 498, 465, 478, 491, 467, 420, 491, 486, 487, 440, 480, 392, 487, 463, 449, 459, 487, 470, 480, 490, 451, 491, 493, 496, 460, 480, 485, 476, 480, 409, 477, 496, 481, 466, 491, 439, 356, 495, 414, 479, 490, 474, 498, 495, 494, 473, 474, 471, 141, 209, 383, 421, 267, 451, 356, 411, 415, 497, 491, 373, 386, 481, 463, 456, 495, 491, 452, 495, 495, 466, 458, 444, 470, 493, 491, 497, 496, 495, 416, 493, 497, 487, 493, 419, 487, 471, 482, 494, 490, 497, 497, 468, 464, 477, 495, 481, 433, 193, 405, 496, 347, 404, 445, 477, 484, 482, 464, 495, 466, 483, 490, 495, 493, 486, 461, 453, 491, 497, 487, 488, 441, 493, 393, 447, 460, 411, 406, 480, 400, 350, 489, 459, 490, 490, 433, 494, 479, 499, 445, 494, 491, 454, 488, 498, 167, 423, 464, 462, 460, 487, 493, 475, 496, 434, 493, 496, 488, 428, 472, 462, 468, 452, 440, 490, 497, 426, 441, 470, 497, 492, 450, 479, 481, 471, 492, 491, 463, 478, 421, 476, 467, 494, 475, 480, 476, 481, 474, 468, 472, 471, 461, 476, 474, 466, 467, 489, 464, 464, 464, 488, 498, 464, 478, 466, 491, 480, 456, 476, 476, 480, 429, 484, 474, 490, 480, 389, 454, 227, 478, 481, 385, 499, 186, 56, 498, 160, 378, 498, 212, 229, 476, 284, 499, 499, 235, 398, 420, 387, 498, 498, 498, 498, 105, 498, 493, 498, 67, 373, 422, 474, 381, 498, 493, 350, 499, 84, 499, 74, 295, 498, 134, 499, 181, 485, 382, 499, 81, 495, 415, 364, 348, 322, 289, 499, 497, 241, 493, 494, 487, 462, 470, 495, 490, 457, 463, 476, 462, 475, 489, 487, 474, 494, 466, 464, 479, 482, 491, 466, 466, 477, 475, 478, 473, 485, 490, 461, 492, 498, 498, 490, 497, 473, 479, 474, 437, 456, 457, 476, 472, 488, 489, 499, 445, 473, 453, 494, 490, 477, 469, 469, 470, 496, 461, 497, 499, 490, 485, 495, 467, 476, 478, 497, 475, 475, 469, 477, 482, 496, 473, 492, 408, 478, 489, 468, 486, 499, 475, 482, 492, 448, 476, 478, 469, 481, 451, 486, 491, 496, 433, 452, 473, 456, 496, 478, 498, 490, 478, 475, 484, 474, 486, 497, 492, 382, 464, 498, 330, 279, 456, 464, 429, 493, 468, 499, 499, 484, 471, 499, 498, 441, 420, 482, 468, 459, 470, 484, 498, 452, 489, 493, 495, 493, 341, 444, 492, 468, 455, 499, 477, 477, 357, 498, 492, 462, 498, 485, 491, 476, 480, 486, 495, 413, 494, 499, 490, 461, 451, 491, 454, 484, 482, 492, 430, 499, 498, 497, 496, 489, 483, 497, 496, 499, 478, 434, 452, 461, 470, 499, 488, 490, 481, 441, 492, 450, 474, 452, 499, 479, 493, 499, 491, 477, 466, 479, 496, 495, 496, 485, 433, 499, 445, 461, 463, 471, 451, 474, 485, 472, 468, 440, 485, 498, 498, 492, 469, 485, 431, 447, 460, 498, 483, 459, 470, 498, 479, 459, 473, 472, 486, 449, 491, 456, 489, 447, 481, 453, 466, 448, 460, 441, 471, 444, 471, 480, 454, 460, 455, 492, 471, 496, 490, 492, 475, 483, 492, 490, 472, 489, 493, 468, 493, 472, 479, 480, 497, 480, 461, 413, 496, 496, 497, 376, 494, 275, 469, 487, 456, 456, 463, 434, 486, 495, 493, 496, 438, 442, 440, 450, 490, 498, 446, 472, 492, 495, 484, 480, 457, 479, 497, 471, 467, 497, 470, 496, 495, 496, 495, 491, 497, 481, 475, 490, 469, 462, 475, 356, 472, 493, 490, 489, 451, 461, 470, 480, 433, 452, 496, 376, 488, 458, 497, 493, 474, 476, 464, 381, 471, 436, 492, 456, 489, 492, 471, 464, 488, 473, 481, 464, 499, 493, 490, 499, 443, 464, 422, 499, 496, 480, 459, 467, 452, 494, 497, 450, 484, 490, 466, 475, 487, 486, 481, 479, 488, 412, 406, 499, 472, 496, 499, 466, 449, 448, 481, 484, 495, 241, 492, 462, 459, 480, 458, 498, 497, 441, 473, 493, 396, 483, 490, 481, 434, 473, 414, 471, 464, 492, 458, 307, 481, 495, 478, 462, 485, 495, 497, 477, 486, 486, 459, 447, 411, 477, 465, 499, 488, 497, 492, 458, 492, 472, 451, 482, 488, 449, 487, 495, 468, 487, 345, 499, 194, 463, 436, 473, 469, 495, 490, 489, 466, 494, 486, 484, 495, 495, 488, 478, 489, 476, 496, 470, 469, 465, 465, 469, 468, 499, 486, 458, 499, 475, 492, 494, 468, 471, 424, 467, 427, 496, 468, 469, 443, 446, 487, 488, 478, 475, 450, 486, 487, 496, 491, 469, 456, 475, 429, 496, 486, 483, 442, 450, 483, 497, 456, 493, 468, 478, 465, 491, 497, 432, 491, 499, 449, 481, 438, 465, 460, 453, 481, 464, 466, 476, 473, 496, 484, 453, 493, 443, 458]，总文字数是 375316.
    print(
        F"分割前，一共有 {len(document_list)} 段，每段文字分别为 {get_document_list_each_length(document_list)}，总文字数是 {get_document_list_length(document_list)}.")
    print(
        F"分割后，一共有 {len(split_document_list)} 段，每段文字分别为 {get_document_list_each_length(split_document_list)}，总文字数是 {get_document_list_length(split_document_list)}.")

    langchain_embeddings = LangchainEmbeddings(embedding_model_path=BGE_LARGE_CN_model_dir)

    # 内部调用 get_or_create_collection，因此不存在会新建，否则沿用
    chroma_db = Chroma(embedding_function=langchain_embeddings,
                       persist_directory=CHROMADB_PATH,
                       collection_name=collection_name,
                       # 指定相似度用内积，支持 cosine, l2, ip
                       collection_metadata={"hnsw:space": "ip"})
    assert_equal(len(chroma_db), 0)

    try:
        chroma_db.add_documents(documents=split_document_list[:20])
        assert_equal(len(chroma_db), 20)

        # llm = CHATGLM4LLM()
        # llm = QWEN2LLM()
        llm = get_langchain_openai_llm(model="gpt-4-turbo", temperature=0.2, real=False)

        memory = ConversationBufferMemory(
            memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
            return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=chroma_db.as_retriever(search_type="similarity", search_kwargs={'k': 3}),
            memory=memory
        )

        question = "我可以学习到关于提示工程的知识吗？"
        result = qa({"question": question})
        print(result, end="\n-----------------------\n")

        question = "为什么这门课需要教这方面的知识？"
        result = qa({"question": question})
        print(result, end="\n-----------------------\n")

        # 可能和 langchain 组织数据的方式有关，目前不能完美的回答上一个问题是什么
        question = "通过上下文，我问的上一个问题是什么，告诉我问题本身即可。"
        result = qa({"question": question})
        print(result, end="\n-----------------------\n")

        # qa = ConversationChain.from_llm(
        #     llm,
        #     # retriever=chroma_db.as_retriever(search_type="similarity", search_kwargs={'k': 3}),
        #     memory=memory
        # )
        #
        # question = "介绍一下你自己"
        # result = qa({"question": question})
        # print(result, end="\n-----------------------\n")
        #
        # question = "我上一个问题是什么？"
        # result = qa({"question": question})
        # print(result, end="\n-----------------------\n")

    finally:
        # 完整删除 collection，不光是 length = 0，而是完全不存在
        chroma_db.delete_collection()


@func_timer(arg=True)
def main():
    assert_equal(len("\n"), 1)
    assert_equal(len("\n\n"), 2)
    assert_equal(len("  \n \n"), 5)

    # check_get_pdf_document_list()
    # check_get_pdf_document_all_in_one()
    # check_get_md_document_list()
    # check_get_md_document_all_in_one()
    # check_get_file_document_list()
    # check_get_file_document_list_all_in_one()
    #
    # check_langchain_embeddings()

    # check_langchain_chroma()

    # check_get_langchain_openai_chat_completion_content()

    # check_langchain_chat_prompt_template()

    # check_langchain_icel()

    # check_custom_chatglm4_llm()

    check_a_whole_rag_demo()
    # check_a_whole_rag_demo_with_memory()


if __name__ == '__main__':
    main()
