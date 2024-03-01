import sys

sys.path.append("../")
from project_utils import *

AIRSIM_FUNCTION_REGEX = re.compile("(aw\\..+\\(?).*\\)?", re.IGNORECASE)


@func_timer(arg=True)
def analysis_command_json(_debug=False):
    all_function_list = ["aw.takeoff", "aw.land", "aw.get_drone_position", "aw.fly_to", "aw.fly_path", "aw.set_yaw",
                         "aw.get_yaw", "aw.get_position"]

    with open("prompt/command.json", "r", encoding="UTF8") as fp:
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

    if len(wrong_function_dict) > 0:
        print(Colors.RED + str(wrong_function_dict) + Colors.ENDC)

    # aw.get_yaw 的调用次数为 0!
    # aw.get_position 的调用次数为 0!
    for function, count in function_count_dict.items():
        if count == 0:
            print(Colors.RED + F"{function} 的调用次数为 0!" + Colors.ENDC)

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

    return question_list, answer_list


def main():
    analysis_command_json(_debug=True)


if __name__ == '__main__':
    main()
