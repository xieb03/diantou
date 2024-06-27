from peft import AutoPeftModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from util_llm import *
from util_path import *


# chatglm 的 chat 函数，内部调用 modeling_chatglm.py 中的 ChatGLMForConditionalGeneration 的 chat 函数
def get_chatglm_completion_content(_tokenizer, _model, user_prompt=None, system_prompt=None,
                                   temperature=0.1, print_cost_time=False, print_response=False,
                                   history_message_list: List = None,
                                   using_history_message_list=True, print_messages=False) -> str:
    start_time = time.time()

    assert user_prompt is None or system_prompt is None, "不支持批量问题，user_prompt 和 system_prompt 只能给一个."

    if user_prompt is not None:
        query = user_prompt
        role = "user"
    else:
        query = system_prompt
        role = "system"

    total_messages = list()
    if using_history_message_list and history_message_list is not None and len(history_message_list) != 0:
        total_messages = history_message_list

    total_messages.append(dict(role=role, content=query))

    if print_messages:
        print("messages:")
        print_history_message_list(total_messages)
        print()

    # 对话记录会自动 extend 到 history_message_list 中，外界可以直接拿到
    # 上面的说法是错误的，因为 chatglm 内部采用了 history = deepcopy(history) 的方式，所以实际上 assistant 的输出没有放到 history 中
    response, history_message_list_copy = _model.chat(_tokenizer, query=query, role=role, temperature=temperature,
                                                      history=history_message_list)
    # 避免上面的问题，将所有的输出都放到历史记录中
    # history_message_list_copy 已经包含了 response
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


# QWEN2 已经集成到了新版的 transformer 中
def get_qwen2_completion_content(_tokenizer: Qwen2TokenizerFast, _model: Qwen2Model, user_prompt=None,
                                 system_prompt=None, temperature=0.1, print_cost_time=False, print_response=False,
                                 history_message_list: List = None, using_history_message_list=True,
                                 print_messages=False) -> str:
    start_time = time.time()

    total_messages = list()
    if using_history_message_list and history_message_list is not None and len(history_message_list) != 0:
        total_messages = history_message_list

    if system_prompt is not None:
        for prompt in to_list(system_prompt):
            total_messages.append(dict(role="system", content=prompt))
            # token_count += len(encoding.encode(prompt))
    if user_prompt is not None:
        for prompt in to_list(user_prompt):
            total_messages.append(dict(role="user", content=prompt))
            # token_count += len(encoding.encode(prompt))

    if print_messages:
        print("messages:")
        print_history_message_list(total_messages)
        print()

    text = _tokenizer.apply_chat_template(
        total_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    generated_ids = _model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192,
        temperature=temperature
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if history_message_list is not None:
        history_message_list.append({"role": "assistant", "content": response})

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
                                     _checkpoint_path=None, _cuda=True, _using_causal_llm=True):
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
        if _cuda:
            chatglm_model = chatglm_model.cuda()
    # 加载原始模型
    else:
        assert is_dir_exist(_pretrained_path), F"文件夹 '{_pretrained_path}' 不存在."
        chatglm_tokenizer, chatglm_model = get_tokenizer_and_model(_pretrained_model_name_or_path=_pretrained_path,
                                                                   _gpu=_cuda,
                                                                   _using_causal_llm=_using_causal_llm)

    return chatglm_model, chatglm_tokenizer


# Loading checkpoint shards: 100%|██████████| 7/7 [00:03<00:00,  2.05it/s]
# model (<class 'transformers_modules.chatglm3-6b.modeling_chatglm.ChatGLMForConditionalGeneration'>) has 6243584032 parameters, 6243584000 (100.00%) are trainable, the dtype is torch.float16，占 11.63G 显存.
# total  gpu memory:  13.82 G
# torch  gpu memory:  11.66 G
# tensor gpu memory:  11.66 G
# C:\Users\admin\.cache\huggingface\modules\transformers_modules\chatglm3-6b\modeling_chatglm.py:226: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at ..\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:455.)
#   context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
# 苏东坡和苏轼是兄弟关系，按照当时的礼制，兄弟之间是不允许举行葬礼的。所以，苏东坡不能参加苏轼的葬礼。
def check_chatglm3(user_prompt):
    model, tokenizer = load_chatglm_model_and_tokenizer(_pretrained_path=CHATGLM3_6B_model_dir)
    print_model_parameter_summary(model)
    print_gpu_memory_summary()

    print(get_chatglm_completion_content(tokenizer, model, user_prompt=user_prompt))


# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# Loading checkpoint shards: 100%|██████████| 10/10 [00:06<00:00,  1.58it/s]
# model (<class 'transformers_modules.glm-4-9b-chat.modeling_chatglm.ChatGLMForConditionalGeneration'>) has 9399951392 parameters, 9399951360 (100.00%) are trainable, the dtype is torch.bfloat16，占 17.51G 显存.
# total  gpu memory:  19.94 G
# torch  gpu memory:  17.6 G
# tensor gpu memory:  17.55 G
# 苏东坡和苏轼实际上是同一个人，苏东坡是苏轼的别称。苏轼，字子瞻，号东坡居士，是北宋时期著名的文学家、书画家、政治家。由于“苏东坡”和“苏轼”是同一个人的不同称呼，所以不存在苏东坡不能参加苏轼葬礼的情况。
#
# 如果这个问题是基于某种特定的历史背景或故事情节，请提供更多的上下文信息，以便给出更准确的答案。在正常情况下，一个人当然可以参加自己的葬礼。
def check_chatglm4(user_prompt):
    model, tokenizer = load_chatglm_model_and_tokenizer(_pretrained_path=GLM4_9B_CHAT_model_dir)
    print_model_parameter_summary(model)
    print_gpu_memory_summary()

    print(get_chatglm_completion_content(tokenizer, model, user_prompt=user_prompt))


# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# Loading checkpoint shards: 100%|██████████| 4/4 [00:04<00:00,  1.10s/it]
# model (<class 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'>) has 7850499328 parameters, 7615616512 (97.01%) are trainable, the dtype is torch.bfloat16，占 14.62G 显存.
# total  gpu memory:  16.15 G
# torch  gpu memory:  14.78 G
# tensor gpu memory:  14.67 G
# 苏东坡和苏轼是同一个人，他们是同一人，即苏轼的别称。苏轼，字子瞻，又字和仲，号铁冠道人、东坡居士，世称苏东坡、苏仙，汉族，北宋眉州眉山（今属四川省眉山市）人，祖籍河北栾城，北宋著名文学家、书法家、画家。
#
# 苏轼一生经历了多次官场沉浮，但他的文学成就和影响力并未因此而减弱。他不仅在文学上有很高的造诣，还对书法、绘画等领域有所贡献。苏轼的去世日期为1083年，而“苏东坡不能参加苏轼的葬礼”这一说法显然是基于混淆了苏轼和苏东坡的身份，实际上，苏东坡就是苏轼，不存在一个叫“苏东坡”的人无法参加自己的葬礼的情况。
def check_qwen2(user_prompt):
    model, tokenizer = load_chatglm_model_and_tokenizer(_pretrained_path=QWEN2_7B_INSTRUCT_model_dir)
    print_model_parameter_summary(model)
    print_gpu_memory_summary()

    print(get_qwen2_completion_content(tokenizer, model, user_prompt=user_prompt))


@func_timer(arg=True)
def main():
    user_prompt = "老鼠生病了，可以吃老鼠药治好么？"
    check_chatglm3(user_prompt)
    check_chatglm4(user_prompt)
    check_qwen2(user_prompt)


if __name__ == '__main__':
    main()
