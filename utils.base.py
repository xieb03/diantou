# 自动实现全属性打印的基类
class Base(object):
    def __str__(self):
        max_attr_length = 0
        for key in self.__dict__.keys():
            max_attr_length = max(max_attr_length, len(key))
        return "\n".join(
            [("%-" + str(max_attr_length) + "s = %s") % (key, value) for key, value in self.__dict__.items()])


class _SubBase(Base):
    a = 3
    bbbbbb = 2

    def __init__(self, a, b):
        self.a = a
        self.bbbbbb = b


def main():
    base = _SubBase(2, 3)
    print(base)


if __name__ == '__main__':
    main()
