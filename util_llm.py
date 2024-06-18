import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from util_plot import *
from util_torch import *


# 获得一个 tokenizer 的 all_special_tokens
def get_tokenizer_all_special_token(_tokenizer: PreTrainedTokenizerFast):
    # [100, 102, 0, 101, 103]
    all_special_ids = _tokenizer.all_special_ids
    # ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    all_special_tokens = _tokenizer.all_special_tokens

    assert_equal(len(all_special_ids), len(all_special_tokens))
    assert is_list_unique(all_special_ids)
    assert is_list_unique(all_special_tokens)

    return dict(zip(all_special_ids, all_special_tokens))


# 从路径中获取 tokenizer, model
# BGE_LARGE_CN_model_dir: {100: '[UNK]', 102: '[SEP]', 0: '[PAD]', 101: '[CLS]', 103: '[MASK]'}
def get_tokenizer_and_model(_pretrained_model_name_or_path=BGE_LARGE_CN_model_dir, _gpu=True):
    # trust_remote_code 表示相信本地的代码，而不是表示同意下载远程代码，不要混淆
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=_pretrained_model_name_or_path,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(_pretrained_model_name_or_path, trust_remote_code=True)
    if _gpu:
        model = model.cuda()

    # print(tokenizer.get_vocab())

    return tokenizer, model


# 获得一句话的 embedding
# _max_length=None，会用所有句子中的最大长度，相当于无限长的上限
def get_sentence_embedding(_tokenizer: PreTrainedTokenizerFast, _model: PreTrainedModel, _sentences, _max_length=None,
                           _gpu=True, _normalize=False):
    # max_length = 1 的时候也会用最长的，等价于 None
    assert _max_length is None or _max_length > 1, "max_length must be greater than 1"
    _model = _model.eval()

    if _max_length is not None and _max_length > 1:
        # 注意要用 batch_encode_plus 而不是 encode，后者只能针对 single 或者 pair，而且会将两句话连成一句话
        input_ids = _tokenizer.batch_encode_plus(_sentences)["input_ids"]
        assert_equal(len(input_ids), len(_sentences))

        for i, input_id in enumerate(input_ids):
            current_length = len(input_id)
            if current_length > _max_length:
                print(
                    F"IMPORTANT: 第 {i} 个句子 '{_sentences[i]}' 的 token 长度 {current_length} 大于 {_max_length}，会被截断.")

    # {'input_ids': tensor([[ 101, 2769,  102],
    #         [ 101, 3299,  102]], device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0],
    #         [0, 0, 0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1],
    #         [1, 1, 1]], device='cuda:0')}
    # 如果 truncation=True，则必须传入 max_length，否则警告：
    # Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
    truncation = True
    if _max_length is None:
        truncation = False
    # 即使传入的 max_length 非常大，大于所有句子的实际 token_count，也会默认用实际情况的最大值来处理，即相当于 max_length 是保护，如果没超就不会额外处理，这样效率最高
    encoded_input_dict = _tokenizer(_sentences, padding=True, truncation=truncation, max_length=_max_length,
                                    return_tensors='pt')
    # print(encoded_input_dict)
    if _gpu:
        _model = _model.cuda()
        change_dict_value_to_gpu(encoded_input_dict)

    with torch.no_grad():
        # BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[-0.0205,  0.0236,  0.0187,  ..., -0.5310, -0.6072,  0.8193],
        #          [ 0.0188,  0.2690,  0.1926,  ...,  0.1011, -0.3511,  0.9317],
        #          [-0.0181,  0.0256,  0.0174,  ..., -0.5313, -0.6066,  0.8208]],
        #
        #         [[ 0.1980,  0.2632, -0.8883,  ..., -0.2285,  0.3321, -0.4319],
        #          [ 0.3331, -0.1477, -0.0881,  ..., -0.3010,  0.2056, -0.1003],
        #          [ 0.2010,  0.2653, -0.8898,  ..., -0.2306,  0.3339, -0.4324]]],
        #        device='cuda:0'), pooler_output=tensor([[ 0.0589, -0.2284, -0.1369,  ...,  0.3764, -0.1964,  0.3106],
        #         [ 0.1524, -0.2249, -0.5382,  ...,  0.2403,  0.2843,  0.0816]],
        #        device='cuda:0'), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
        model_output = _model(**encoded_input_dict)
        # Perform pooling. In this case, cls pooling.
        # cls-pooling：直接取 [CLS] 的 embedding
        # mean-pooling：取每个 Token 的平均 embedding
        # max-pooling：对得到的每个 Embedding 取 max
        # tensor([[-0.0205,  0.0236,  0.0187,  ..., -0.5310, -0.6072,  0.8193],
        #         [ 0.1980,  0.2632, -0.8883,  ..., -0.2285,  0.3321, -0.4319]],
        #        device='cuda:0')
        sentence_embeddings = model_output[0][:, 0]

    if _normalize:
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


# 不考虑 tokenizer，只是判断 words 是不是在 tokenizer 的 vocab 中
# 而如果考虑 tokenizer，即使是一个 word，也可能会被分为多个 token，即这个 word 不再 vocab 中，但这多个 token 都在 vocab 中
# 即严格意义上来说这个方法是不准确的，即没有 word 的 embedding，只有 token 的 embedding
# 而且 word 和 token 即使是一堆野，也不一定是明文文对应的，例如在 QWEN2_7B_INSTRUCT_model_dir 中，"月"对应的 token 是 æľĪ，id 是 9754
# 即 input_ids = tokenizer.encode("月") = [9754]，而 对应的 vocab 中 value = 9754 的 key 是 æľĪ
# tokenizer.convert_ids_to_tokens(tokenizer.encode_plus('月', truncation=True, max_length=5,
#                                 padding="max_length", return_token_type_ids=True, return_special_tokens_mask=True)['input_ids']) = ['æľĪ', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>']
# 注意 tokenizer.encode() 是对一个字符串进行 encoder，即使一个字符串只有一个字，也可能会有多个 tokens.
def get_word_embeddings(_vocab_or_tokenizer: Union[Dict[str, int], PreTrainedTokenizerFast], _model: PreTrainedModel,
                        _words):
    # 注意不能用 encode，因为形如 "##月" 实际上是在 vocab 中，但会被当做 3 个字来处理
    # assert_equal(len(_words), 1)
    # BGE_LARGE_CN_model_dir: {100: '[UNK]', 102: '[SEP]', 0: '[PAD]', 101: '[CLS]', 103: '[MASK]'}
    # input_ids = _tokenizer.encode(_words)
    # bos_token_id = _tokenizer.bos_token_id or _tokenizer.cls_token_id
    # eos_token_id = _tokenizer.eos_token_id or _tokenizer.sep_token_id
    # assert_equal(input_ids[0], bos_token_id)
    # assert_equal(input_ids[-1], eos_token_id)
    # # print(_tokenizer.get_vocab())
    # print(_tokenizer.get_vocab()[_words])
    # if len(input_ids) > 3:
    #     return None
    # token_id = input_ids[1]
    if isinstance(_vocab_or_tokenizer, dict):
        vocab = _vocab_or_tokenizer
    else:
        vocab = _vocab_or_tokenizer.get_vocab()
    input_id = vocab.get(_words, None)
    if input_id is None:
        return None
    else:
        # clone 提供了非数据共享的梯度追溯功能，在不共享数据内存的同时支持梯度回溯，所以常用在神经网络中某个单元需要重复使用的场景下。
        # detach “舍弃”了梯度功能，在共享数据内存的脱离计算图，所以常用在神经网络中仅要利用张量数值，而不需要追踪导数的场景下
        # clone 和 detach 意味着着只做简单的数据复制，既不数据共享，也不对梯度共享，从此两个张量无关联。
        # torch.nn.modules.sparse.Embedding
        # 等价于下面的算法，因为 Embedding 对象有 __call__(input)的方法
        # 不能用 Tensor 而要用 tensor，因为 Tensor 传入一个整数 n 时，torch.Tensor 认识 n 是一维张量的元素个数，并随机初始化
        # input_id = torch.tensor(input_id)
        # if is_model_on_gpu(_model):
        #     input_id = input_id.cuda()
        # return _model.get_input_embeddings()(input_id).clone().detach()

        return _model.get_input_embeddings().weight[input_id].clone().detach()


def check_word_embeddings():
    tokenizer, model = get_tokenizer_and_model(BGE_LARGE_CN_model_dir)
    # tensor([ 0.0180,  0.0267, -0.0128,  ...,  0.0132, -0.0190, -0.0698],
    #        device='cuda:0')
    # tensor([-0.0406,  0.0038,  0.0193,  ...,  0.0341,  0.0010,  0.0383],
    #        device='cuda:0')
    # None
    # tensor([ 0.0060,  0.0471, -0.0275,  ...,  0.0521,  0.0461, -0.0111],
    #        device='cuda:0')
    print(get_word_embeddings(tokenizer, model, "√"))
    print(get_word_embeddings(tokenizer, model, "月"))
    print(get_word_embeddings(tokenizer, model, "#月"))
    print(get_word_embeddings(tokenizer, model, "##月"))


def check_sentence_embeddings():
    sentences = ["我爱我老婆陈平", "月"]
    tokenizer, model = get_tokenizer_and_model(BGE_LARGE_CN_model_dir)
    # 即使传入的 max_length 非常大，大于所有句子的实际 token_count，也会默认用实际情况的最大值来处理，即相当于 max_length 是保护，如果没超就不会额外处理，这样效率最高
    # max_length = None 和 _max_length = 1 和 max_length = 20 是一样的，都会用最长来标记
    # tensor([[-0.5532,  0.8763, -0.4879,  ...,  0.3610, -0.3822,  0.4251],
    #         [ 0.1980,  0.2632, -0.8883,  ..., -0.2285,  0.3321, -0.4319]],
    #        device='cuda:0')
    # IMPORTANT: 第 0 个句子 '我爱我老婆陈平' 的 token 长度 9 大于 3，会被截断.
    # tensor([[-0.0205,  0.0236,  0.0187,  ..., -0.5310, -0.6072,  0.8193],
    #         [ 0.1980,  0.2632, -0.8883,  ..., -0.2285,  0.3321, -0.4319]],
    #        device='cuda:0')
    # tensor([[-0.5532,  0.8763, -0.4879,  ...,  0.3610, -0.3822,  0.4251],
    #         [ 0.1980,  0.2632, -0.8883,  ..., -0.2285,  0.3321, -0.4319]],
    #        device='cuda:0')
    print(get_sentence_embedding(tokenizer, model, sentences))
    print(get_sentence_embedding(tokenizer, model, sentences, _max_length=3))
    print(get_sentence_embedding(tokenizer, model, sentences, _max_length=200))


def check_sentence_compare():
    in_1 = "He could not desert his post at the power plant."
    in_2 = "The power plant needed him at the time."
    in_3 = "Desert plants can survive droughts."
    in_4 = "This is a meaningful sentence."

    input_text_lst_sim = [in_1, in_2, in_3, in_4]
    tokenizer, model = get_tokenizer_and_model(BGE_LARGE_EN_model_dir)

    embeddings = []
    for input_text in input_text_lst_sim:
        emb = get_sentence_embedding(tokenizer, model, input_text)
        embeddings.append(emb[0])

    n_sample = len(input_text_lst_sim)

    # (3, 1024)\
    # numpy() 必须在 cpu() 之后才能调用，不能从 cuda 直接到 numpy()
    embeddings_array = np.array([embedding.cpu().numpy() for embedding in embeddings])
    assert_equal(embeddings_array.shape[0], n_sample)

    # Perform PCA for 2D visualization
    pca_model = PCA(n_components=2)
    pca_array = pca_model.fit_transform(embeddings_array)
    # [[-8.0531845 -1.8658072]
    #  [-6.2139134 -0.4001754]
    #  [ 9.216008  -6.952848 ]
    #  [ 5.0510926  9.21883  ]]
    print(pca_array)

    # perplexity must be less than n_samples
    tsne_model = TSNE(n_components=2, init="pca", perplexity=1)
    tsne_array = tsne_model.fit_transform(embeddings_array)
    # [[-17.89582   388.9884   ]
    #  [-10.780518  288.0105   ]
    #  [ 11.957385  -23.07555  ]
    #  [  0.5574764 132.70337  ]]
    print(tsne_array)

    # n_neighbors must be greater than 1
    umap_model = umap.UMAP(n_neighbors=2, n_components=2, n_jobs=1, random_state=0, transform_seed=42)
    umap_array = umap_model.fit_transform(embeddings_array)
    # [[ 5.2945724 11.356611 ]
    #  [ 6.2063622 11.4417   ]
    #  [ 6.613435  10.035056 ]
    #  [ 6.8994374 10.681623 ]]
    print(umap_array)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=pca_array[:, 0], y=pca_array[:, 1], s=200, label="pca")
    sns.scatterplot(x=tsne_array[:, 0], y=tsne_array[:, 1], s=200, label="tsne")
    sns.scatterplot(x=umap_array[:, 0], y=umap_array[:, 1], s=200, label="umap")

    for i in range(n_sample):
        plt.text(pca_array[i, 0], pca_array[i, 1], str(i), fontsize=20)
        plt.text(tsne_array[i, 0], tsne_array[i, 1], str(i), fontsize=20)
        plt.text(umap_array[i, 0], umap_array[i, 1], str(i), fontsize=20)
    plt.show()

    def compare(idx1, idx2):
        return F.cosine_similarity(embeddings[idx1], embeddings[idx2], dim=0)

    # He could not desert his post at the power plant.
    # The power plant needed him at the time.
    # tensor(0.8405, device='cuda:0')
    print(in_1)
    print(in_2)
    print(compare(0, 1))

    # He could not desert his post at the power plant.
    # Desert plants can survive droughts.
    # tensor(0.4408, device='cuda:0')
    print(in_1)
    print(in_3)
    print(compare(0, 2))


@func_timer(arg=True)
def main():
    fix_all_seed(_simple=False)

    # check_word_embeddings()
    # check_sentence_embeddings()
    check_sentence_compare()


if __name__ == '__main__':
    main()
