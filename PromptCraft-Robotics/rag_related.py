import sys

from sklearn.model_selection import train_test_split

sys.path.append("../")
from project_utils import *

# 不支持嵌套，只支持多个单个函数串联
# AIRSIM_FUNCTION_REGEX = re.compile("(aw\\..+\\(?).*\\)?", re.IGNORECASE)

# 支持 aw.fly_to(aw.get_position('car')) 的嵌套形式，会输出 ['aw.fly_to', 'aw.get_position'] 两个函数
AIRSIM_FUNCTION_REGEX = re.compile("(aw\\..+?\\(+?)", re.IGNORECASE)
CHROMADB_COLLECTION_NAME = "normal"


# 从文件中获得 question_list, answer_list
def get_question_answer_list_from_file(_command_json_path_or_command_list, _is_file=True):
    if _is_file:
        assert is_file_exist(_command_json_path_or_command_list), F"文件 '{_command_json_path_or_command_list} 不存在."
        with open(_command_json_path_or_command_list, "r", encoding="UTF8") as fp:
            command_list = json.load(fp)
    else:
        command_list = _command_json_path_or_command_list

    length = len(command_list)
    # 一共有 100 条数据.
    print(F"一共有 {length // 2} 条数据.")

    question_list = list()
    answer_list = list()

    for i in range(0, length, 2):
        user_command = command_list[i]
        assert user_command["role"] == "user", F"第 {i // 2} 段并不是 user."
        question = user_command["content"]
        question_list.append(question)
        assistant_command = command_list[i + 1]
        assert assistant_command["role"] == "assistant", F"第 {(i + 1) // 2} 段并不是 assistant."
        answer = assistant_command["content"]
        # 注意 answer 包括整个 content，而不仅仅是 python 代码
        answer_list.append(answer)

    return question_list, answer_list


# 分析 command.json 的一些统计值
def analysis_command_json(question_list=None, answer_list=None, _command_json_path="prompt/command.json", _debug=False):
    if question_list is None or answer_list is None:
        question_list, answer_list = get_question_answer_list_from_file(_command_json_path, _is_file=True)

    all_function_list = ["aw.takeoff", "aw.land", "aw.get_drone_position", "aw.fly_to", "aw.fly_path", "aw.set_yaw",
                         "aw.get_yaw", "aw.get_position"]

    function_count_in_each_command_dict = defaultdict(int)
    function_count_dict = dict.fromkeys(all_function_list, 0)
    # sub_function_count_dict = defaultdict(int)
    # function_count_detail_dict = {function: list() for function in all_function_list}
    wrong_function_dict = defaultdict(int)

    for question, answer in zip(question_list, answer_list):
        # 优先用三引号寻找代码
        python_code = extract_python_code(answer)
        # 如果没有找到，那么直接按照 python 代码处理
        if not python_code:
            python_code = answer.strip()

        # 检查 python 代码是否能编译通过，注意这里只检查代码的合理性
        assert check_python_code_syntax_error(python_code), (
                Colors.RED + f"获取的 python 代码编译有问题，请重新试过: {python_code}" + Colors.ENDC)

        function_list = AIRSIM_FUNCTION_REGEX.findall(python_code)
        function_count = len(function_list)
        function_count_in_each_command_dict[function_count] += 1

        for function in function_list:
            main_function = function[:function.index("(")]
            if main_function not in function_count_dict:
                wrong_function_dict[main_function] += 1
            else:
                function_count_dict[main_function] += 1
                # sub_function_count_dict[function] += 1
                # function_count_detail_dict[main_function].append(function)

    function_count_in_each_command_dict = dict(sorted(function_count_in_each_command_dict.items(),
                                                      key=operator.itemgetter(1), reverse=True))
    function_count_dict = dict(sorted(function_count_dict.items(),
                                      key=operator.itemgetter(1), reverse=True))
    # sub_function_count_dict = dict(sorted(sub_function_count_dict.items(),
    #                                       key=lambda x: (x[1], x[0]), reverse=True))
    # function_count_detail_dict = dict(sorted(function_count_detail_dict.items(),
    #                                          key=lambda x: len(x[1]), reverse=True))
    # for value in function_count_detail_dict.values():
    #     value.sort()

    # question_list 和 answer_list 去重
    question_dict = defaultdict(int)
    answer_dict = defaultdict(int)
    for question in question_list:
        question_dict[question] += 1
    for answer in answer_list:
        answer_dict[answer] += 1

    repeat_question_dict = {key: count for key, count in question_dict.items() if count > 1}
    repeat_answer_dict = {key: count for key, count in answer_dict.items() if count > 1}
    repeat_question_dict = dict(sorted(repeat_question_dict.items(), key=operator.itemgetter(1), reverse=True))
    repeat_answer_dict = dict(sorted(repeat_answer_dict.items(), key=operator.itemgetter(1), reverse=True))

    final_question_list = list()
    final_answer_list = list()
    for i in range(len(question_list)):
        question = question_list[i]
        answer = answer_list[i]
        if question not in repeat_question_dict and answer not in repeat_answer_dict:
            final_question_list.append(question)
            final_answer_list.append(answer)
    assert len(final_question_list) == len(final_answer_list)

    if _debug:
        # 每个 answer 中含有的 function 的个数的分布
        # key = 个数，value = 频次
        # 5: 66
        # 4: 20
        # 6: 13
        # 2: 1
        print("-" * 80)
        print_dict(function_count_in_each_command_dict)
        print("-" * 80)
        # aw.fly_to: 184
        # aw.set_yaw: 124
        # aw.takeoff: 97
        # aw.get_drone_position: 43
        # aw.land: 26
        # aw.fly_path: 16
        # aw.get_yaw: 0
        # aw.get_position: 0

        # 主函数的调用次数
        print_dict(function_count_dict)
        print("-" * 80)

        # # 子函数（含参）的调用次数
        # print_dict(sub_function_count_dict)
        # print("-" * 80)

        # # 每个主函数中的所有子函数
        # print_dict(function_count_detail_dict)
        # print("-" * 80)

    if len(wrong_function_dict) > 0:
        print(Colors.RED + str(wrong_function_dict) + Colors.ENDC)

    # aw.get_yaw 的调用次数为 0!
    # aw.get_position 的调用次数为 0!
    for function, count in function_count_dict.items():
        if count == 0:
            print(Colors.RED + F"{function} 的调用次数为 0!" + Colors.ENDC)

    if len(repeat_question_dict) > 0:
        for (key, value) in repeat_question_dict.items():
            print(Colors.RED + F"question '{key}' 出现了 {value} 次." + Colors.ENDC)
    if len(repeat_answer_dict) > 0:
        for (key, value) in repeat_answer_dict.items():
            print(Colors.RED + F"answer '{key}' 出现了 {value} 次." + Colors.ENDC)

    # 去重后，共有 37 条有效记录.
    print(F"去重后，共有 {len(final_question_list)} 条有效记录.")

    # for question, answer in zip(final_question_list, final_answer_list):
    #     print(str({"role": "user", "content": question}) + ",")
    #     print(str({"role": "assistant", "content": answer}) + ",")

    return final_question_list, final_answer_list


# 将 command.json 中的 QA pair 对插入 chromadb 向量数据库中
@func_timer(arg=False)
def insert_pre_prompt_to_chromadb(_command_json_path="prompt/command.json", _collection_name=CHROMADB_COLLECTION_NAME,
                                  _debug=True,
                                  _top_n=3):
    question_list, answer_list = analysis_command_json(_command_json_path=_command_json_path, _debug=_debug)
    assert len(question_list) == len(answer_list)

    collection = ChromadbPersistentCollection(collection_name=_collection_name)
    # total  gpu memory:  6.38 G
    # torch  gpu memory:  4.54 G
    # tensor gpu memory:  4.52 G
    print_gpu_memory_summary()

    collection.reset()

    # 80% 作为训练集，20% 作为测试集
    question_list_train, question_list_test, answer_list_train, answer_list_test = (
        train_test_split(question_list, answer_list, test_size=0.2, random_state=0))

    collection.add(
        documents=question_list_train,
        uris=answer_list_train
    )

    if _debug:
        for i in range(min(len(question_list_test), 5)):
            question = question_list_test[i]
            answer = answer_list_test[i]
            get_rag_results(_question=question, _answer=answer, _collection=collection, _debug=_debug, _top_n=_top_n)

    # collection.drop()


# 根据问题从 chromadb 向量数据库中找到几个相似的 QA 对，这里采用召回和排序的方法，最多返回 top_n 对
@func_timer(arg=False)
def get_rag_results(_question, _answer=None, _collection=None, _collection_name=CHROMADB_COLLECTION_NAME, _debug=True,
                    _top_n=3, _min_similarity=0.4):
    if _collection is None:
        _collection = ChromadbPersistentCollection(collection_name=_collection_name)

    all_similar_answer = _collection.query_and_rank_one(_question)
    questions = all_similar_answer["documents"]
    similarities = all_similar_answer["similarities"]
    scores = all_similar_answer["scores"]
    similar_answers = all_similar_answer["uris"]

    if _debug:
        print(Colors.RED + "-" * 80)
        print("DEBUG")
        question = _question.replace("\n", "\n")
        print(F"test_question = {question}")
        if _answer is not None:
            _answer = _answer.replace("\n", "\n")
            print(F"test_answer = ```{_answer}```")
        print()
        answer_count = len(questions)
        for j in range(min(answer_count, _top_n)):
            print(F"\tsimilarity = {similarities[j]:.4F}")
            print(F"\tscore = {scores[j]:.4F}")
            questions[j] = questions[j].replace("\n", "\n\t\t")
            print(F"\tsimilar_question = {questions[j]}")
            similar_answers[j] = similar_answers[j].replace("\n", "\n\t\t")
            print(F"\tanswer = ```{similar_answers[j]}```")
            print()
        print("-" * 80 + Colors.ENDC)

    rag_answer_list = similar_answers[:_top_n]
    rag_question_list = questions[:_top_n]
    origin_rag_count = len(rag_answer_list)

    if _min_similarity is not None:
        rag_similarity_list = similarities[:_top_n]
        new_rag_question_list = list()
        new_rag_answer_list = list()
        for question, answer, similarity in zip(rag_question_list, rag_answer_list, rag_similarity_list):
            if similarity >= _min_similarity:
                new_rag_question_list.append(question)
                new_rag_answer_list.append(answer)
        rag_count = len(new_rag_question_list)
        if rag_count < origin_rag_count:
            print(Colors.RED + F"用相似度阈值 {_min_similarity:.2F} 只保留了 {rag_count} 条记录." + Colors.ENDC)

        return new_rag_question_list, new_rag_answer_list
    else:
        return rag_question_list, rag_answer_list


# 将 rag 结果组装成最终 prompt
def assemble_prompt_from_template(_question, _rag_question_list, _rag_answer_list, _prompt_template):
    count = len(_rag_question_list)
    assert count == len(_rag_answer_list)
    records = ""
    for i in range(count):
        records += "user: " + _rag_question_list[i] + "\n"
        records += "assistant: ```" + _rag_answer_list[i] + "```\n"
        records += "\n"

    return _prompt_template.format(question=_question, count=count, records=records)


# 将 command.json 转化为 lora 微调的格式
def covert_command_to_lora(_command_json_path: str, _fix_json_path: str, _debug: bool = True):
    question_list, answer_list = analysis_command_json(_command_json_path=_command_json_path, _debug=_debug)
    assert len(question_list) == len(answer_list)

    # 80% 作为训练集，20% 作为测试集
    question_list_train, question_list_test, answer_list_train, answer_list_test = (
        train_test_split(question_list, answer_list, test_size=0.2, random_state=0))

    for split, sub_question_list, sub_answer_list in zip(["train", "dev"], [question_list_train, question_list_test],
                                                         [answer_list_train, answer_list_test]):
        count = 0
        with open(_fix_json_path.format(split=split), 'wt', encoding='utf-8') as fout:
            for question, answer in zip(sub_question_list, sub_answer_list):
                count += 1
                sample = {'conversations': [{'role': 'user', 'content': question},
                                            {'role': 'assistant', 'content': answer}]}
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(F"{split} 一共保存了 {count} 条数据.")


@func_timer(arg=True)
def main():
    fix_all_seed()

    # analysis_command_json(_command_json_path="prompt/command_old.json", _debug=True)
    # analysis_command_json(_command_json_path="prompt/command.json", _debug=True)
    # insert_pre_prompt_to_chromadb(_command_json_path="prompt/command.json", _debug=True)
    #
    # question = "向前飞 100 米."
    # rag_question_list, rag_answer_list = get_rag_results(_question=question, _debug=True)
    # prompt_template = ("You will be shown a new question, and some relevant historical dialogue records. "
    #                    "You can refer to these records but should use "
    #                    "the actual distance, direction, and coordinates from the new question. "
    #                    "If you feel that the relevant records are unreasonable, "
    #                    "you can ignore them, but tell me the reasons."
    #                    "\n\nnew question: {question}\n\n{count} relevant records: \n{records}")
    #
    # print(assemble_prompt_from_template(question, rag_question_list, rag_answer_list, prompt_template))

    # 去重后，共有 130 条有效记录.
    # train 一共保存了 104 条数据.
    # dev 一共保存了 26 条数据.
    covert_command_to_lora(_command_json_path="prompt/command.json", _fix_json_path="prompt/command_fix_{split}.json")


if __name__ == '__main__':
    main()
