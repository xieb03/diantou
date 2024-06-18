from modelscope import snapshot_download

from util_path import *


# 如果之前下载过，会通过判断文件是否存在的方式，如果都存在则不会继续下载
# 模型的目录在 BIGDATA_MODELS_PATH 中，而具体名称由 modelscope 自己指定，例如 ZhipuAI/chatglm3-6b 的目录就是 ZhipuAI\chatglm3-6b
# revision 是 git 的分支名，因此不能知道当前模型到底是哪个分支的，或者说，一旦某一个分支已经下载了，其它的分支就不能更新，除非将当前已经保存的模型删除

# 2024-06-18 15:12:14,142 - modelscope - INFO - PyTorch version 2.3.0+cu121 Found.
# 2024-06-18 15:12:14,145 - modelscope - INFO - Loading ast index from C:\Users\admin\.cache\modelscope\ast_indexer
# 2024-06-18 15:12:14,266 - modelscope - INFO - Loading done! Current index file version is 1.15.0, with md5 5cb81a06bdbd703e8cd3eaa3065e1fa4 and a total number of 980 components indexed
# 2024-06-18 15:12:16,537 - modelscope - INFO - Use user-specified model revision: v1.0.0
def modelscope_download(model_id=CHATGLM3_6B_model_id, revision=CHATGLM3_6B_model_revision, local_files_only=False):
    # D:\PycharmProjects\xiebo\diantou\bigdata\models\ZhipuAI\chatglm3-6b
    model_dir = snapshot_download(model_id=model_id, revision=revision, cache_dir=BIGDATA_MODELS_PATH,
                                  local_files_only=local_files_only)
    print(model_dir)
    # assert_equal(model_dir, get_model_dir(model_id=model_id))

    return model_dir


# 并不对所有的都适用
# "AI-ModelScope/bge-large-zh-v1.5" -> "AI-ModelScope/bge-large-zh-v1___5"
def get_model_dir(model_id=CHATGLM3_6B_model_id):
    model_dir = PATH_SEPARATOR.join(multiply_split([",", "\\", "/"], delete_all_blank(model_id)))
    return BIGDATA_MODELS_PATH + model_dir


@func_timer(arg=True)
def main():
    assert_equal(modelscope_download(model_id=CHATGLM3_6B_model_id, revision=CHATGLM3_6B_model_revision),
                 CHATGLM3_6B_model_dir)
    assert_equal(modelscope_download(model_id=BGE_LARGE_CN_model_id, revision=BGE_LARGE_CN_model_revision),
                 BGE_LARGE_CN_model_dir)
    assert_equal(modelscope_download(model_id=BGE_RERANKER_LARGE_model_id, revision=BGE_RERANKER_LARGE_revision),
                 BGE_RERANKER_LARGE_model_dir)
    assert_equal(modelscope_download(model_id=BGE_LARGE_EN_model_id, revision=BGE_LARGE_EN_model_revision),
                 BGE_LARGE_EN_model_dir)
    assert_equal(modelscope_download(model_id=GLM4_9B_CHAT_model_id, revision=GLM4_9B_CHAT_model_revision),
                 GLM4_9B_CHAT_model_dir)
    assert_equal(modelscope_download(model_id=QWEN2_7B_INSTRUCT_model_id, revision=QWEN2_7B_INSTRUCT_model_revision),
                 QWEN2_7B_INSTRUCT_model_dir)


if __name__ == '__main__':
    main()
