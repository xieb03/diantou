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
        elif len(str(_timestamp)) == 16:
            _timestamp = _timestamp // 1000000
        elif len(str(_timestamp)) == 19:
            _timestamp = _timestamp // 1000000000
        if len(str(_timestamp)) != 10:
            raise ValueError(F"只支持 10(秒)、13(毫秒)、16(微妙)、19(纳秒)，不支持 {_timestamp}, length={len(str(_timestamp))}")
    else:
        raise ValueError(type(_timestamp))

    return datetime.datetime.fromtimestamp(_timestamp)


def get_current_datetime_str(_style="%Y%m%d-%H%M%S"):
    timestamp = get_current_timestamp()
    return change_datetime_to_str(get_datetime_from_timestamp(timestamp), _style=_style)


# 获得当前时间戳，支持 秒、毫秒、微妙、纳秒
# 注意 time.time 的小数点后只有 7 位
def get_current_timestamp(_digit=10):
    if _digit == 10:
        return int(time.time())
    elif _digit == 13:
        return int(time.time() * 1000)
    elif _digit == 16:
        return int(time.time() * 1000000)
    elif _digit == 19:
        return int(time.time() * 1000000000)
    raise ValueError(F"只支持 10(秒)、13(毫秒)、16(微妙)、19(纳秒)，不支持 {_digit}")


# 获得秒级时间戳
def get_current_timestamp_second():
    return get_current_timestamp(10)


# 获得毫秒级时间戳
def get_current_timestamp_millisecond():
    return get_current_timestamp(13)


# 获得微秒级时间戳
def get_current_timestamp_microsecond():
    return get_current_timestamp(16)


# 获得纳秒级时间戳
def get_current_timestamp_nanosecond():
    return get_current_timestamp(19)


def main():
    pass


if __name__ == '__main__':
    main()
