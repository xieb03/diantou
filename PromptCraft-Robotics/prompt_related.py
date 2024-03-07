import sys

sys.path.append("../")
from project_utils import *

# 不支持嵌套，只支持多个单个函数串联
# AIRSIM_FUNCTION_REGEX = re.compile("(aw\\..+\\(?).*\\)?", re.IGNORECASE)

# 支持 aw.fly_to(aw.get_position('car')) 的嵌套形式，会输出 ['aw.fly_to', 'aw.get_position'] 两个函数
AIRSIM_FUNCTION_REGEX = re.compile("(aw\\..+?\\(+?)", re.IGNORECASE)
CHROMADB_COLLECTION_NAME = "normal"


def get_prompt_from_chatgpt():
    with open("prompt/system_prompt.txt", "r", encoding="UTF8") as f:
        system_prompt = f.read()

    with open("prompt/user_prompt.txt", "r", encoding="UTF8") as f:
        user_prompt = f.read()

    history_message_list = list()

    response = get_chatgpt_completion_content(user_prompt=user_prompt, system_prompt=system_prompt,
                                              history_message_list=history_message_list)

    print(response)


@func_timer(arg=True)
def main():
    get_prompt_from_chatgpt()


if __name__ == '__main__':
    main()
