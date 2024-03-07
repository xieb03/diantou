from peft import AutoPeftModelForCausalLM
from transformers import AutoModel, AutoTokenizer

from util_path import *
from util_spider import *


# chatglm 的 chat 函数，内部调用 modeling_chatglm.py 中的 ChatGLMForConditionalGeneration 的 chat 函数
def get_chatglm_completion_content(_tokenizer, _model, user_prompt=None, system_prompt=None,
                                   temperature=0.1, print_cost_time=False, print_response=False,
                                   history_message_list: List = None,
                                   using_history_message_list=True, print_messages=False):
    start_time = time.time()

    assert user_prompt is None or system_prompt is None, "不支持批量问题，user_prompt 和 system_prompt 只能给一个."

    if user_prompt is not None:
        query = user_prompt
        role = "user"
    else:
        query = system_prompt
        role = "system"

    if print_messages:
        total_messages = list()
        if using_history_message_list and history_message_list is not None and len(history_message_list) != 0:
            total_messages = history_message_list

        total_messages.append(dict(role=role, content=query))

        print("messages:")
        print_history_message_list(total_messages)
        print()

    # 对话记录会自动 extend 到 history_message_list 中，外界可以直接拿到
    # 上面的说法是错误的，因为 chatglm 内部采用了 history = deepcopy(history) 的方式，所以实际上 assistant 的输出没有放到 history 中
    response, history_message_list_copy = _model.chat(_tokenizer, query=query, role=role, temperature=temperature,
                                                      history=history_message_list)
    # 避免上面的问题，将所有的输出都放到历史记录中
    if history_message_list is not None:
        history_message_list.clear()
        history_message_list.extend(history_message_list_copy)

    if print_response:
        print(response)

    end_time = time.time()
    cost_time = end_time - start_time

    if print_cost_time:
        print(F"cost time = {cost_time:.1F}s.")
        print()

    return response


# 加载 chatglm 模型，可以选择原始模型或者微调后的 checkpoint
# 加载微调模型可能遇到的问题
# property 'eos_token' of 'ChatGLMTokenizer' object has no setter
# 将 checkpoint-400/tokenizer_config.json 中的 eos_token 去掉
# property 'pad_token' of 'ChatGLMTokenizer' object has no setter
# 将 checkpoint-400/tokenizer_config.json 中的 pad_token 去掉
# property 'unk_token' of 'ChatGLMTokenizer' object has no setter
# 将 checkpoint-400/tokenizer_config.json 中的 unk_token 去掉
# File "D:\Users\admin\anaconda3\Lib\site-packages\transformers\modeling_utils.py", line 1585, in set_input_embeddings
#     base_model.set_input_embeddings(value)
#   File "D:\Users\admin\anaconda3\Lib\site-packages\transformers\modeling_utils.py", line 1587, in set_input_embeddings
#     raise NotImplementedError
# 在源模型的 modeling_chatglm.py 中 ChatGLMModel 类第 770 行 加入以下代码
#     def set_input_embeddings(self, value):
#         self.embedding.word_embeddings = value
def load_chatglm_model_and_tokenizer(_pretrained_path=CHATGLM3_6B_model_dir, _use_checkpoint=False,
                                     _checkpoint_path=None, _cuda=True):
    # 是否用某个 checkpoint
    use_checkpoint = _use_checkpoint and _checkpoint_path is not None
    # 加载 checkpoint
    if use_checkpoint:
        assert is_dir_exist(_checkpoint_path), F"文件夹 '{_checkpoint_path}' 不存在."
        adapter_config_path = keep_one_end_path_separator(_checkpoint_path) + 'adapter_config.json'
        assert is_file_exist(adapter_config_path), F"文件 '{adapter_config_path}' 不存在."
        chatglm_model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=_checkpoint_path, trust_remote_code=True, device_map='auto'
        ).eval()
        tokenizer_dir = chatglm_model.peft_config['default'].base_model_name_or_path
        chatglm_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_dir, trust_remote_code=True
        )
    # 加载原始模型
    else:
        assert is_dir_exist(_pretrained_path), F"文件夹 '{_pretrained_path}' 不存在."
        chatglm_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=_pretrained_path,
                                                          trust_remote_code=True)
        chatglm_model = AutoModel.from_pretrained(pretrained_model_name_or_path=_pretrained_path,
                                                  trust_remote_code=True).eval()
    if _cuda:
        chatglm_model = chatglm_model.cuda()

    return chatglm_model, chatglm_tokenizer


def main():
    pass


if __name__ == '__main__':
    main()
