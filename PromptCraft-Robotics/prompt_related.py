from rag_related import *

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
                                              history_message_list=history_message_list,
                                              model="gpt-4")
    print(response)

    user_prompt = ('情况有一些变化，我需要基于上面的指令和场景，来生成一些历史对话记录，这些对话中将作为未来任务的记录参考。'
                   '在这些记录里，你将扮演我的角色发出指令，然后根据这个指令进行回复。'
                   # '记录格式为 "{"role": "user", "content": "指令"}, {"role": "assistant", "content": "回复内容，包含完整的 python 代码"}。'
                   '每个记录都包括指令和回复两部分，记录格式为 "{"conversations":[{"role": "user", "content": "指令"}, {"role": "assistant", "content": "回复，仅包含 python 代码"}]}。'
                   '注意对话记录中的指令要是明确没有歧义的，例如不要单独说左侧或者上方，而要包括具体的坐标或者方位，这样回复内容才能根据对应的指令进行精确的答复而不需要额外假设，而且回复不能包含任何额外的函数，必须是之前给定的。'
                   # '指令一定不能重复，而且要尽量简单，使得回复不要有太复杂的函数调用。'
                   '指令和回复一定不能重复，可以包含简单的指令，例如 "向前飞 10 米", "飞到 tower1", "朝向顺时针旋转 90 度" 等，也可以更加复杂或者嵌套，例如 "飞到 tower1 前面 10 米的地方并获得位置"，"途经 tower1, tower3, car 再降落"，"环绕 tower2 飞一个半径为 10 米的圆形" 等。'
                   '不要局限在我给的例子中，你可以加入更多的元素。一共生成 20 组记录，在这些记录中，要覆盖尽量多的场景，使得回复中对于函数的调用更均匀。以 JSON 格式输出，不要包含其它的内容。')

    response = get_chatgpt_completion_content(user_prompt=user_prompt,
                                              history_message_list=history_message_list, temperature=1.2,
                                              model="gpt-4")
    print(response)

    if "```json" in response:
        json_blocks = JSON_CODE_BLOCK_REGEX.findall(response)
        assert len(json_blocks) == 1, F"{len(json_blocks)}, {json_blocks}"
        print("-" * 80)
        print(json_blocks[0])
        print("-" * 80)
        command_list = json.loads(json_blocks[0])['conversations']
    else:
        command_list = json.loads(response)['conversations']
    for command in command_list:
        print(str(command) + ",")

    # get_question_answer_list_from_file(command_list)
    # json_blocks = JSON_CODE_BLOCK_REGEX.findall(response)
    # assert len(json_blocks) == 1, F"{len(json_blocks)}, {json_blocks}"
    # print(json_blocks[0])

    question_list, answer_list = get_question_answer_list_from_file(command_list, _is_file=False)
    analysis_command_json(question_list, answer_list, _debug=True)


@func_timer(arg=True)
def main():
    get_prompt_from_chatgpt()


if __name__ == '__main__':
    main()
