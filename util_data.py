import datasets
# noinspection PyUnresolvedReferences
import huggingface_hub
from datasets import load_dataset

from util_path import *


# 下载数据集，数据集必须在 https://huggingface.co/api/datasets 中，可以用下列 python 脚本查看
# print_list(huggingface_hub.list_datasets(dataset_name="scan"))

# name: config_name 或者是子数据集的名称
# 注意会在 BIGDATA_DATA_PATH 中额外生成 downloads 文件夹，目前来看并不影响
# split: 如何划分数据集
#       If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
#       If given, will return a single Dataset.
# keep_in_memory: 是否保存在内存中，默认是 None，即如果设置了环境变量 IN_MEMORY_MAX_SIZE，那么数据集的大小小于这个值即可以放入内存，否则为 False
def dataset_download(path, name=None, split=None, keep_in_memory=None, _return=True, _print=False, _info=False):
    if _info:
        # 设置日志级别是 INFO，默认是 WARNING
        datasets.logging.set_verbosity_info()

    dataset = load_dataset(path=path, name=name, split=split, keep_in_memory=keep_in_memory,
                           cache_dir=BIGDATA_DATA_PATH)
    if _print:
        print(dataset)

    if _return:
        return dataset


@func_timer(arg=True)
def main():
    # print_list(huggingface_hub.list_datasets(dataset_name="scan"))

    # DatasetDict({
    #     train: Dataset({
    #         features: ['commands', 'actions'],
    #         num_rows: 16728
    #     })
    #     test: Dataset({
    #         features: ['commands', 'actions'],
    #         num_rows: 4182
    #     })
    # })
    # dataset_download(path="scan", name="simple", _print=True, _info=True)

    pass


if __name__ == '__main__':
    main()
