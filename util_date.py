import datetime

from util import *


# 将 datetime.date 或者datetime.datetime 类型转化为指定样式的字符串
# 注意文件名里面不能有 [":"]
def change_datetime_to_str(_datetime, _style="%Y%m%d-%H%M%S"):
    return _datetime.strftime(_style)


# 从时间戳获得datetime
def get_datetime_from_timestamp(_timestamp):
    if isinstance(_timestamp, (int, float, str, np.int64, np.int32)):
        try:
            _timestamp = int(_timestamp)
        except ValueError:
            raise ValueError(_timestamp)

        if len(str(_timestamp)) == 13:
            _timestamp = _timestamp // 1000
        if len(str(_timestamp)) != 10:
            raise ValueError(_timestamp)
    else:
        raise ValueError(type(_timestamp))

    return datetime.datetime.fromtimestamp(_timestamp)


def main():
    pass


if __name__ == '__main__':
    main()
