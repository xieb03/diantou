import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from util_torch import *


# BGE_LARGE_CN 的 EmbeddingFunction
class BGELargeCNEmbeddingFunction(EmbeddingFunction):
    # noinspection PyProtocol
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

    def __call__(self, input_: Documents) -> Embeddings:
        encoded_input = self.embedding_tokenizer(input_, padding=True, truncation=True, max_length=self.max_length,
                                                 return_tensors='pt')
        if self.using_gpu:
            change_dict_value_to_gpu(encoded_input)

        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            # cls-pooling：直接取 [CLS] 的 embedding
            # mean-pooling：取每个 Token 的平均 embedding
            # max-pooling：对得到的每个 Embedding 取 max
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        return torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).tolist()


# BGE_LARGE_CN 的 EmbeddingFunction
class BGEReRankingFunction:
    # noinspection PyProtocol
    def __init__(self, rerank_model_path=BGE_RERANKER_LARGE_model_dir, max_length=1024, using_gpu=True):
        self.rank_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=rerank_model_path,
            trust_remote_code=True)
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=rerank_model_path,
            trust_remote_code=True)
        self.using_gpu = using_gpu
        if self.using_gpu:
            self.rank_model = self.rank_model.cuda()
        self.max_length = max_length

    def __call__(self, one: Documents, others: Documents) -> List[float]:
        # 凑成 pair 对，相当于 batch
        pairs = list()
        for another in to_list(others):
            pairs.append([one, another])
        with torch.no_grad():
            # 注意 padding = True 必须指定，因为这里相当于是 batch，必须将所有样本凑成一样长的，否则报错
            encoded_input = self.rank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt',
                                                max_length=self.max_length)
            # 如果 model 在显卡中，那么参数也要都在显卡中
            if self.using_gpu:
                change_dict_value_to_gpu(encoded_input)
            scores = self.rank_model(**encoded_input, return_dict=True).logits.view(-1, ).float().tolist()

        return scores


class ChromadbPersistentCollection:

    # 如果只是进行 query 而不进行 add，可以指定 embedding_function=None，节省显存
    # 上面的话不对，因为 query 也要进行 embedding 的运算
    def __init__(self, persistent_path=CHROMADB_PATH, collection_name="my_collection", metadata=None,
                 embedding_function=BGELargeCNEmbeddingFunction(BGE_LARGE_CN_model_dir),
                 reranker=BGEReRankingFunction(BGE_RERANKER_LARGE_model_dir)):
        self.client = chromadb.PersistentClient(path=persistent_path)

        self.metadata = metadata
        if self.metadata is None:
            # l2, ip, cosine
            self.metadata = {"hnsw:space": "ip"}
        self.name = collection_name
        self.embedding_function = embedding_function
        self._get_or_create_collection()

        self.reranker = reranker

    # 获取 collection
    def _get_or_create_collection(self):
        self.collection = self.client.get_or_create_collection(name=self.name, metadata=self.metadata,
                                                               embedding_function=self.embedding_function)

    def count(self):
        return self.collection.count()

    # 清空 collection，collection 还存在，只是没有数据，即 count == 0
    def reset(self):
        self.drop()
        self._get_or_create_collection()
        assert self.count() == 0

    # 删除 collection，collection 不再存在
    def drop(self):
        self.client.delete_collection(self.name)

    # 增加多条记录
    def add(self, documents, ids=None, uris=None, print_count=True):
        documents = to_list(documents)
        if ids is not None:
            ids = to_list(ids)
        if uris is not None:
            uris = to_list(uris)

        # 注意 ids 不能为空，而且不要有重复，这里用数据条数作为 id，保证不重复
        if ids is None:
            count = self.count()
            ids = list()
            for i in range(len(documents)):
                ids.append(str(count + i))

        self.collection.add(
            documents=documents,
            ids=ids,
            uris=uris
        )

        if print_count:
            print(F"{self.name} 一共有 {self.count()} 条数据.")

    # noinspection PyTypedDict
    def query_one(self, query_text, n_results=6, include=None, simplify_result=True):
        # 避免警告：Number of requested results 5 is greater than number of elements in index 2, updating n_results = 2
        n_results = min(n_results, self.count())
        if include is None:
            include = ["documents", "distances", "uris"]

        # 注意 distance = 1 - 相似度，即距离越小，相似度越大
        results = self.collection.query(query_texts=query_text, n_results=n_results, include=include)

        if simplify_result:
            simple_results = dict()
            simple_results["ids"] = list(map(int, results["ids"][0]))
            simple_results["documents"] = results["documents"][0]

            for key in ["uris", "embeddings", "metadatas", "data"]:
                if results[key] is not None:
                    simple_results[key] = results[key][0]

            # 为了查看方便，将距离转化为相似度输出
            if results["distances"] is not None:
                simple_results["similarities"] = list(map(lambda x: round(1 - x, 4), results["distances"][0]))

            return simple_results
        else:
            return results

    # noinspection PyTypedDict
    def query_and_rank_one(self, query_text, n_results=6, include=None, simplify_result=True):
        if self.reranker is None:
            raise ValueError("必须指定 reranker.")
        results = self.query_one(query_text=query_text, n_results=n_results, include=include,
                                 simplify_result=simplify_result)
        results["scores"] = self.reranker(query_text, results["documents"])
        assert len(results["scores"]) == len(results["ids"])

        sort_dict_with_one_value_list(results, _key="scores", _sub_key="similarities", _reverse=True)
        results["scores"] = list(map(lambda x: round(x, 4), results["scores"]))

        return results


@func_timer(arg=True)
def main():
    # 1024 [0.001481, 0.01648, -0.028145, 0.025962, 0.012677]
    # 1024 [0.015082, 0.004146, -0.015681, 0.03677, 0.017891]
    # 1024 [0.011628, 0.005764, -0.008518, 0.009795, 0.041958]
    embedding_function = BGELargeCNEmbeddingFunction(BGE_LARGE_CN_model_dir)
    for embedding in embedding_function(["样例数据-1", "样例数据-2", "错例数据-2"]):
        print(len(embedding), list(map(get_round_6, to_list(embedding)[:5])))

    collection = ChromadbPersistentCollection(collection_name="test", embedding_function=embedding_function)
    # total  gpu memory:  6.38 G
    # torch  gpu memory:  4.54 G
    # tensor gpu memory:  4.52 G
    print_gpu_memory_summary()

    collection.reset()

    collection.add(
        documents=["This is a document", "This is the third document", "This is another document"],
        uris=["details of 0", "details of 2", "details of 1"]
    )

    # {'ids': [0, 2, 1], 'documents': ['This is a document', 'This is another document',
    # 'This is the third document'], 'uris': ['details of 0', 'details of 1', 'details of 2'],
    # 'similarities': [1.0, 0.8707, 0.8097]}
    print(collection.query_one("This is a document"))

    # {'ids': (0, 2, 1), 'documents': ('This is a document', 'This is another document', 'This is the third document'),
    # 'uris': ('details of 0', 'details of 1', 'details of 2'), 'similarities': (1.0, 0.8707, 0.8097),
    # 'scores': [9.4713, 3.3706, 0.3262]}
    print(collection.query_and_rank_one("This is a document"))

    collection.drop()


if __name__ == '__main__':
    main()
