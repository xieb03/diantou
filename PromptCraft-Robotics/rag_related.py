import sys

from sklearn.model_selection import train_test_split

sys.path.append("../")
from project_utils import *

AIRSIM_FUNCTION_REGEX = re.compile("(aw\\..+\\(?).*\\)?", re.IGNORECASE)
CHROMADB_COLLECTION_NAME = "normal"


@func_timer(arg=False)
def analysis_command_json(_command_json_path="prompt/command.json", _debug=False):
    all_function_list = ["aw.takeoff", "aw.land", "aw.get_drone_position", "aw.fly_to", "aw.fly_path", "aw.set_yaw",
                         "aw.get_yaw", "aw.get_position"]

    with open(_command_json_path, "r", encoding="UTF8") as fp:
        command_list = json.load(fp)

    length = len(command_list)
    # 一共有 100 条数据.
    print(F"一共有 {length // 2} 条数据.")

    question_list = list()
    answer_list = list()

    function_count_in_each_command_dict = defaultdict(int)
    function_count_dict = dict.fromkeys(all_function_list, 0)
    sub_function_count_dict = defaultdict(int)
    function_count_detail_dict = {function: list() for function in all_function_list}
    wrong_function_dict = defaultdict(int)
    for i in range(0, length, 2):
        user_command = command_list[i]
        assert user_command["role"] == "user"
        question = user_command["content"]
        question_list.append(question)
        assistant_command = command_list[i + 1]
        assert assistant_command["role"] == "assistant"
        answer = assistant_command["content"]
        # 注意 answer 包括整个 content，而不仅仅是 python 代码
        answer_list.append(answer)

        python_code = extract_python_code(answer)
        function_list = AIRSIM_FUNCTION_REGEX.findall(python_code)
        function_count = len(function_list)
        function_count_in_each_command_dict[function_count] += 1

        for function in function_list:
            main_function = function[:function.index("(")]
            if main_function not in function_count_dict:
                wrong_function_dict[main_function] += 1
            else:
                function_count_dict[main_function] += 1
                sub_function_count_dict[function] += 1
                function_count_detail_dict[main_function].append(function)

    function_count_in_each_command_dict = dict(sorted(function_count_in_each_command_dict.items(),
                                                      key=operator.itemgetter(1), reverse=True))
    function_count_dict = dict(sorted(function_count_dict.items(),
                                      key=operator.itemgetter(1), reverse=True))
    sub_function_count_dict = dict(sorted(sub_function_count_dict.items(),
                                          key=lambda x: (x[1], x[0]), reverse=True))
    function_count_detail_dict = dict(sorted(function_count_detail_dict.items(),
                                             key=lambda x: len(x[1]), reverse=True))
    for value in function_count_detail_dict.values():
        value.sort()

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
        # aw.fly_to: 184
        # aw.set_yaw: 124
        # aw.takeoff: 97
        # aw.get_drone_position: 43
        # aw.land: 26
        # aw.fly_path: 16
        # aw.get_yaw: 0
        # aw.get_position: 0
        print("-" * 80)
        # 主函数的调用次数
        print_dict(function_count_dict)
        print("-" * 80)
        # 子函数（含参）的调用次数
        print_dict(sub_function_count_dict)
        print("-" * 80)
        # 每个主函数中的所有子函数
        print_dict(function_count_detail_dict)
        print("-" * 80)

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

    return final_question_list, final_answer_list


@func_timer(arg=True)
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


@func_timer(arg=False)
def get_rag_results(_question, _answer=None, _collection=None, _collection_name=CHROMADB_COLLECTION_NAME, _debug=True,
                    _top_n=3):
    if _collection is None:
        _collection = ChromadbPersistentCollection(collection_name=_collection_name)

    all_similar_answer = _collection.query_and_rank_one(_question)
    questions = all_similar_answer["documents"]
    similarities = all_similar_answer["similarities"]
    scores = all_similar_answer["scores"]
    similar_answers = all_similar_answer["uris"]

    rag_answer_list = similar_answers[:_top_n]
    rag_question_list = questions[:_top_n]

    if _debug:
        question = _question.replace("\n", "\n\t")
        print(Colors.GREEN + F"test_question = {question}" + Colors.ENDC)
        if _answer is not None:
            _answer = _answer.replace("\n", "\n\t")
            print(Colors.GREEN + F"test_answer = {_answer}" + Colors.ENDC)
        answer_count = len(questions)
        for j in range(min(answer_count, _top_n)):
            print(Colors.BLUE + F"\tsimilarity = {similarities[j]:.4F}" + Colors.ENDC)
            print(Colors.BLUE + F"\tscore = {scores[j]:.4F}" + Colors.ENDC)
            questions[j] = questions[j].replace("\n", "\n\t\t")
            print(Colors.BLUE + F"\tsimilar_question = {questions[j]}" + Colors.ENDC)
            similar_answers[j] = similar_answers[j].replace("\n", "\n\t\t")
            print(Colors.BLUE + F"\tanswer = {similar_answers[j]}" + Colors.ENDC)
        print("-" * 80)

    return rag_question_list, rag_answer_list


def main():
    fix_all_seed()

    # analysis_command_json(_debug=True)
    insert_pre_prompt_to_chromadb(_debug=True)


if __name__ == '__main__':
    main()
