from modelscope import snapshot_download

from util_path import *


# 如果之前下载过，会通过判断文件是否存在的方式，如果都存在则不会继续下载
# 模型的目录在 BIGDATA_MODELS_PATH 中，而具体名称由 modelscope 自己指定，例如 ZhipuAI/chatglm3-6b 的目录就是 ZhipuAI\chatglm3-6b
# revision 是 git 的分支名，因此不能知道当前模型到底是哪个分支的，或者说，一旦某一个分支已经下载了，其它的分支就不能更新，除非将当前已经保存的模型删除
def modelscope_download(model_id=CHATGLM3_6B_model_id, revision=CHATGLM3_6B_model_revision, local_files_only=False):
    # D:\PycharmProjects\xiebo\diantou\bigdata\models\ZhipuAI\chatglm3-6b
    model_dir = snapshot_download(model_id=model_id, revision=revision, cache_dir=BIGDATA_MODELS_PATH,
                                  local_files_only=local_files_only)
    assert_equal(model_dir, get_model_dir(model_id=model_id))


def get_model_dir(model_id=CHATGLM3_6B_model_id):
    model_dir = PATH_SEPARATOR.join(multiply_split([",", "\\", "/"], delete_all_blank(model_id)))
    return BIGDATA_MODELS_PATH + model_dir


def main():
    modelscope_download(model_id=CHATGLM3_6B_model_id, revision=CHATGLM3_6B_model_revision)


if __name__ == '__main__':
    main()
