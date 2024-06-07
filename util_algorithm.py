from util import *

# 计算 24 点用到的所有运算顺序，包括加减乘除，另外因为减法和触法不具备交换性，因此实际上有 6 种运算。
# 使用 itertools.product() 生成所有操作符的排列，注意是可以重复的排列，因此是 6 * 6 * 6 = 216 而不是 A(6, 3) = 120
OPERATION_COMBINATIONS_LIST = list(itertools.product(["+", "-", "*", "/", "-r", "/r"], repeat=3))
assert is_list_unique(OPERATION_COMBINATIONS_LIST)
assert_equal(len(OPERATION_COMBINATIONS_LIST), 6 ** 3)


# 计算 24 点的所有结果
def get_24_results(_list: List[int]) -> List[str]:
    def get_expression(_a, _b, _op) -> str:
        if _op == "+":
            return "({} + {})".format(_a, _b)
        elif _op == "-":
            return "({} - {})".format(_a, _b)
        elif _op == "-r":
            return "({} - {})".format(_b, _a)
        elif _op == "*":
            return "({} * {})".format(_a, _b)
        elif _op == "/":
            return "({} / {})".format(_a, _b)
        elif _op == "/r":
            return "({} / {})".format(_b, _a)
        else:
            raise ValueError(f"Unknown operation: {_op}")

    for value in _list:
        assert isinstance(value, int), "{} is not int.".format(value)

    assert_equal(len(_list), 4)

    # 使用 itertools.permutations() 生成数字的排列，注意如果有重复元素 permutations 并不会去重，因此这里人工去一次重
    permutations_list = sorted(list(set(itertools.permutations(_list))))
    assert len(permutations_list) <= 24

    expression_list = list()
    for permutations in permutations_list:
        for operations in OPERATION_COMBINATIONS_LIST:
            expression = None
            for i, operation in enumerate(operations):
                if i == 0:
                    expression = get_expression(permutations[0], permutations[1], operation)
                else:
                    expression = get_expression(expression, permutations[i + 1], operation)
            # noinspection PyBroadException
            try:
                if np.isclose(eval(expression), 24):
                    expression_list.append(expression)
                    # print(expression, permutations, operations)
            except Exception:
                continue
    # 注意可能会有重复，例如 (1, 2, 3, 4) * (/, /, /r) 和 (2, 1, 3, 4) * ('/r', '/', '/r') 结果是相同的
    expression_list = delete_repeat_value_and_keep_order(expression_list)
    return expression_list


def main():
    assert_equal(len(get_24_results([1, 2, 3, 4])), 66)
    assert_equal(len(get_24_results([4, 4, 10, 10])), 1)


if __name__ == '__main__':
    main()
