import calendar
import datetime

from util import *

__DEFAULT_DATETIME_STR_STYLE = "%Y-%m-%d %H:%M:%S"
__DEFAULT_DATE_STR_STYLE = "%Y-%m-%d"
__DEFAULT_MONTH_STR_STYLE = "%Y-%m"

# 时间戳样式
DATETIME_STR_STYLE = "%Y/%m%d %H:%M:%S"
DATE_STR_STYLE = "%Y/%m%d"
TIME_STR_STYLE = "%H:%M:%S"


# 根据起始时间戳构造所有的分钟级的间隔
def get_all_datetime_minute_time_slice_list_from_datetime(_start_datetime, _end_datetime, _slice_range=5,
                                                          _tz=datetime.timezone.utc,
                                                          _strftime="%Y%m%d%H%M", _assert_strict=True):
    start_time_slice: int = get_datetime_minute_time_slice(_start_datetime, _slice_range, _tz, _strftime)
    end_time_slice: int = get_datetime_minute_time_slice(_end_datetime, _slice_range, _tz, _strftime)

    return get_all_datetime_minute_time_slice_list_from_time_slice(start_time_slice, end_time_slice, _slice_range, _tz,
                                                                   _strftime, _assert_strict)


# 根据起始时间戳构造所有的分钟级的间隔
def get_all_datetime_minute_time_slice_list_from_time_slice(_start_time_slice: int, _end_time_slice: int,
                                                            _slice_range: int = 5,
                                                            _tz: datetime.timezone = datetime.timezone.utc,
                                                            _strftime: str = "%Y%m%d%H%M",
                                                            _assert_strict: bool = True) -> List[int]:
    all_datetime_minute_time_slice_list = [_start_time_slice]
    current_time_slice = _start_time_slice
    while True:
        current_time_slice = add_minute_on_minute_time_slice(current_time_slice, _slice_range, _tz, _strftime=_strftime)
        all_datetime_minute_time_slice_list.append(current_time_slice)
        if current_time_slice > _end_time_slice:
            break

        if _assert_strict:
            assert _end_time_slice in all_datetime_minute_time_slice_list

    return all_datetime_minute_time_slice_list


# 验证分钟级 time_slice_list 是合理的，每一个都是 slice_range 的倍数，而且是连续不间断的
def check_datetime_minute_time_slice_list(_datetime_minute_time_slice_list, _slice_range,
                                          _strftime="%Y%m%d%H%M") -> bool:
    for time_slice in _datetime_minute_time_slice_list:
        assert (time_slice % 60) % _slice_range == 0, F"{(time_slice % 60) % _slice_range} != 0"

    for i in range(len(_datetime_minute_time_slice_list) - 1):
        assert get_two_minute_time_slice_gap(_datetime_minute_time_slice_list[i],
                                             _datetime_minute_time_slice_list[i + 1], _strftime) == _slice_range

    assert_equal(
        get_two_minute_time_slice_gap(_datetime_minute_time_slice_list[0], _datetime_minute_time_slice_list[-1],
                                      _strftime, _slice_range), len(_datetime_minute_time_slice_list) - 1)

    return True


# 将时间戳化为分钟级的间隔 _slice_range 的时间片
# 一定要注意时区的设置
# 这个时间片是可以比较大小的
def get_datetime_minute_time_slice(_datetime, _slice_range=5, _tz=datetime.timezone.utc, _strftime="%Y%m%d%H%M",
                                   _extend_minutes_list=None) -> Union[float, int, List[int]]:
    if pd.isnull(_datetime):
        return np.nan

    duration_gap = _slice_range * 60
    timestamp_minute = int(_datetime.timestamp() / duration_gap) * duration_gap

    # <class 'datetime.datetime'>
    datetime_slice = datetime.datetime.fromtimestamp(timestamp_minute, _tz)
    time_slice = int(datetime_slice.strftime(_strftime))

    if _extend_minutes_list is None:
        return time_slice

    result_list = [time_slice]
    for extend_minute in to_list(_extend_minutes_list):
        result_list.append(int((datetime_slice + datetime.timedelta(minutes=extend_minute)).strftime(_strftime)))

    return result_list


# 将分钟级的间隔时间片 标签（12位整数） 增加 _add_value 分钟
# 一定要注意时区的设置
def add_minute_on_minute_time_slice(_time_slice_int: int, _add_minute: int,
                                    _tz: datetime.timezone = None,
                                    _strftime: str = "%Y%m%d%H%M") -> int:
    assert 1e11 < _time_slice_int < 1e12, _time_slice_int

    timestamp_minute = datetime.datetime.strptime(str(_time_slice_int), _strftime).timestamp() + _add_minute * 60
    return int(datetime.datetime.fromtimestamp(timestamp_minute, _tz).strftime(_strftime))


# 获取两个分钟级间隔时间片之间的分钟差
# 后面的减前面的
def get_two_minute_time_slice_gap(_time_slice_int_1: int, _time_slice_int_2: int, _strftime: str = "%Y%m%d%H%M",
                                  _slice_range: int = 1) -> float:
    timestamp_1 = datetime.datetime.strptime(str(_time_slice_int_1), _strftime).timestamp()
    timestamp_2 = datetime.datetime.strptime(str(_time_slice_int_2), _strftime).timestamp()
    return (timestamp_2 - timestamp_1) / 60 / _slice_range


# 判断一个 date_list 是否是连续的，这里可以事先不排好序
def is_date_str_list_continuous(_date_list: List[str], _date_str_style: str = __DEFAULT_DATE_STR_STYLE) -> bool:
    if not is_list_unique(_date_list):
        return False

    min_date = min(_date_list)
    max_date = max(_date_list)
    return len(_date_list) == get_date_diff(min_date, max_date, _date_str_style=_date_str_style, _include=True)


# 判断一个字符串是否是正确的时间格式，注意，不光是格式，不合理的时间也会报错，例如 20241313
def is_str_right_datetime_style(_str: str, _date_str_style: str = DATE_STR_STYLE):
    # noinspection PyBroadException
    try:
        get_date_str_from_datetime(from_datetime_str_to_datetime(_str, _date_str_style), _date_str_style)
        return True
    except Exception:
        return False


# 根据 datetime 计算整数 hour
def get_hour(_datetime: datetime.datetime) -> int:
    # data["tsd_dttm].dt.hour
    return int(_datetime.strftime("%H"))


# 根据 datetime 计算整数 minute
def get_minute(_datetime: datetime.datetime) -> int:
    # data["tsd_dttm].dt.minute
    return int(_datetime.strftime("%M"))


# 是否是 python 标准
def is_python_weekday(_python_weekday: int) -> bool:
    return 0 <= _python_weekday <= 6


# 是否是 java 标准
def is_java_weekday(_java_weekday: int) -> bool:
    return 1 <= _java_weekday <= 7


# java 标准：周一 = 2，周六 = 7，周日 = 1
# python 标准：周一 = 1，周六 = 6，周日 = 0
# 从 python 标准转为 java 标准
def get_java_weekday_from_python_weekday(_python_weekday: int) -> int:
    assert is_python_weekday(_python_weekday), _python_weekday
    return _python_weekday + 1


# java 标准：周一 = 2，周六 = 7，周日 = 1
# python 标准：周一 = 1，周六 = 6，周日 = 0
# 从 java 标准转为 python 标准
def get_python_weekday_from_java_weekday(_java_weekday: int) -> int:
    assert is_java_weekday(_java_weekday), _java_weekday
    return _java_weekday - 1


# 尝试将一个 dataframe 中所有形如 datetime 的列全部转化
def try_pd_to_datetime(_df: pd.DataFrame, _format: str = "%Y-%m-%d %H:%M:%S.%f") -> List[str]:
    _format = to_list(_format)
    datetime_column_list = list()
    for column in _df.columns:
        column_dtypes = _df[column].dtypes
        # 只有字符串型才考虑转化成时间戳
        # old: 不包含整型，因为整型都能被转化，切转化结果很可能不对
        # update: 增加 from_datetime_str_to_datetime 校验后，整型也可以加进来
        # 不包含浮点型，因为浮点类型大概率不是
        if column_dtypes == "object" or "int" in str(column_dtypes):
            # 注意 [column] 要放在前面，否则先取 iloc[0] 的时候会因为是 series，将会使得所有的类型强转，例如 int -> float
            # one_sample = _df[~_df[column].isnull()].iloc[0][column]
            one_sample = _df[~_df[column].isnull()][column].iloc[0]
            for current_format in _format:
                # noinspection PyBroadException
                try:
                    # 因为 pd.to.datetime 的检查不严格，因此用 from_datetime_str_to_datetime 严格检查
                    # assert str(pd.to_datetime("2022-06-06 21-10", format="%Y-%m-%d")) == "2022-06-06 21:00:00-10:00"
                    # # from_datetime_str_to_datetime("2022-06-06 21-10","%Y-%m-%d")
                    # assert (str(pd.to_datetime("2022-06-06 21:56:35.929", format="%Y-%m-%d")) ==
                    #         "2022-06-06 21:56:35.929000")
                    # assert str(from_datetime_str_to_datetime("2022-06-06 21:56:35.929",
                    #                                          "%Y-%m-%d %H:%M:%S.%f")) == "2022-06-06 21:56:35.929000"
                    # from_datetime_str_to_datetime("2022-06-06 21:56:35.929", "%Y-%m-%d")
                    # assert str(pd.to_datetime(20220606, format="%Y-%m-%d")) == "1970-01-01 99:00:00.020220606"
                    # assert str(pd.to_datetime(20220606, format="%Y%m%d")) == "2022-06-06 00:00:00"
                    # assert str(from_datetime_str_to_datetime("20220606", format="%Y%m%d")) == "2022-06-06 00:00:00"
                    # # from_datetime_str_to_datetime("20220606", "%Y-%m-%d")
                    from_datetime_str_to_datetime(str(one_sample), current_format)
                    pd.to_datetime(one_sample, format=current_format)
                    _df[column] = pd.to_datetime(_df[column], format=current_format)
                    datetime_column_list.append(column)
                    print("{} 是时间戳，格式是 {}".format(column, current_format))
                    break
                except Exception:
                    print("{} 不是时间戳".format(column))
                    pass
    print("一共有 {} 个列是时间戳，分别是 {}".format(len(datetime_column_list), datetime_column_list))
    return datetime_column_list


# 包含时间的 print
def print_with_time(_msg: str) -> None:
    print("{}:{}".format(get_now_datetime_str("%Y-%m-%d %H:%M:%S"), _msg))


# 获得当前时区，注意北京是 -8，韩国是 -9
def get_timezone() -> int:
    return int(time.timezone / 3600)


# 获得当前日期时间字符串
def get_now_datetime_str(_datetime_str_style="%Y-%m-%d %H:%M:%S") -> str:
    return time.strftime(_datetime_str_style, time.localtime(time.time()))


# 获得当前日期字符串
def get_now_date_str(_date_str_style="%Y-%m-%d") -> str:
    return time.strftime(_date_str_style, time.localtime(time.time()))


# 获得当前时间字符串
def get_now_time_str(_time_str_style="%H:%M:%S") -> str:
    return time.strftime(_time_str_style, time.localtime(time.time()))


# 根据时间间隔获得天、时、分、秒，最多支持到天，注意传入的是秒为单位的
def get_time_interval_str(_seconds_interval) -> str:
    sign = "-" if _seconds_interval < 0 else ""
    is_int = isinstance(_seconds_interval, int)
    _seconds_interval = abs(_seconds_interval)
    milliseconds = _seconds_interval - math.floor(_seconds_interval)
    _seconds_interval = math.floor(_seconds_interval)
    seconds = _seconds_interval % 60
    _seconds_interval -= seconds
    minutes = (_seconds_interval % 3600) // 60
    _seconds_interval -= minutes * 60
    hours = (_seconds_interval % 86400) // 3600
    _seconds_interval -= hours * 3600
    days = _seconds_interval // 86400

    if is_int:
        result = str(seconds) + " seconds"
    else:
        result = str(math.floor(milliseconds * 1000)) + " milliseconds"
        if seconds != 0:
            result = str(seconds) + " seconds " + result
    if minutes != 0:
        result = str(minutes) + " minutes " + result
    if hours != 0:
        result = str(hours) + " hours " + result
    if days != 0:
        result = str(days) + " days " + result
    return sign + result


# 给定起始日期和终止日期，获得日期整形列表
def get_date_list_by_start_and_end(_start_date: str, _end_date: str, _date_str_style=DATETIME_STR_STYLE, _to_int=False) \
        -> List[Union[int, str]]:
    date_list = [datetime.datetime.strftime(x_, _date_str_style) for x_ in list(pd.date_range(_start_date, _end_date))]
    if _to_int:
        date_list = [int(date) for date in date_list]
    return date_list


# 通过 date_list 或者起始日期来确定最终的 date_list，优先以 date_list 为准，如果 _end_date 是空的话，截止到当天
def get_date_list_by_list_or_by_start_and_end(_date_list: List[str] = None, _start_date=None, _end_date=None,
                                              _date_str_style: str = "%Y%m%d") -> Union[None, List[str]]:
    if _date_list is not None:
        return to_list(_date_list)
    if _start_date is not None and _end_date is not None:
        return get_date_list_by_start_and_end(_start_date, _end_date, _date_str_style)
    if _start_date is not None and _end_date is None:
        return get_date_list_by_start_and_end(_start_date, get_today(_date_str_style), _date_str_style)
    return None


# 获得当前日期
def get_today(_date_str_style: str = __DEFAULT_DATE_STR_STYLE) -> str:
    return datetime.datetime.today().strftime(_date_str_style)


# 获得昨天日期
def get_yesterday(_date_str_style: str = __DEFAULT_DATE_STR_STYLE) -> str:
    return get_date_with_delta_days(get_today(_date_str_style), -1, _date_str_style)


# 获得两个月份的差
def get_month_diff(_start_month: str, _end_month: str, _month_str_style=__DEFAULT_MONTH_STR_STYLE) -> int:
    start_month: datetime.datetime = datetime.datetime.strptime(_start_month, _month_str_style)
    end_month: datetime.datetime = datetime.datetime.strptime(_end_month, _month_str_style)
    return 12 * (end_month.year - start_month.year) + (end_month.month - start_month.month)


# 当 datetime 字符串转化为 datetime 类型
def from_datetime_str_to_datetime(_datetime_str: str, _datetime_str_style: str = DATE_STR_STYLE) -> datetime.datetime:
    return datetime.datetime.strptime(_datetime_str, _datetime_str_style)


# datetime.datetime.strftime(datetime, _style) 将 datetime 类型转化为字符串
# datetime.datetime.strptime(datetime_str, _style) 将字符串类型转化为 datetime
def get_date_str_from_datetime(_datetime: datetime.datetime, _date_str="%Y-%m-%d") -> str:
    return datetime.datetime.strftime(_datetime, _date_str)


# 获得某个日期向前、向后的日期
def get_date_with_delta_days(_date: Union[str, int, datetime.datetime], delta_days: int,
                             _date_str_style: str = __DEFAULT_DATE_STR_STYLE) -> str:
    if isinstance(_date, (str, int)):
        _date: datetime.datetime = from_datetime_str_to_datetime(str(_date), _date_str_style)
    return get_date_str_from_datetime(_date + datetime.timedelta(days=delta_days), _date_str_style)


# 改变时间字符的样式
def change_time_str_style(_time_str: str, _in_style: str, _out_style: str) -> str:
    return get_date_str_from_datetime(from_datetime_str_to_datetime(_time_str, _in_style), _out_style)


# 获得 datetime 的星期几
# 统一采用 python 标准
# python 的 datetime.strftime("%w") 默认周日是 0，周六是 6
# java 的 Calendar.DAY_OF_WEEK 默认周日是 1，周六是 7
def get_weekday_by_datetime(_datetime: datetime.datetime) -> int:
    return int(_datetime.strftime("%w"))


# 获得一个时间戳最近的上一个星期几，如果时间戳是星期五，那么最近的星期五是上一个星期五，不包括当天
# _include 表示是否包含当天，上面的情况如果包括当天，那么就直接返回当天
def get_nearest_weekday_datetime_by_datetime(_datetime: datetime.datetime, _weekday: int,
                                             _include: bool = False) -> datetime.datetime:
    # 0 ~ 6，周日是 0，周六是 6
    assert is_python_weekday(_weekday)
    # 注意周日到周六分别是 0 到 6
    current_weekday = get_weekday_by_datetime(_datetime)
    delta_days = current_weekday - _weekday
    if delta_days == 0:
        if not _include:
            delta_days = -7
    else:
        if delta_days > 0:
            delta_days = -delta_days
        else:
            delta_days = -(7 + delta_days)
    new_datetime = _datetime + datetime.timedelta(days=delta_days)
    assert_equal(get_weekday_by_datetime(new_datetime), _weekday)
    return new_datetime


def get_nearest_weekday_datetime_by_timestamp(_timestamp: Union[int, float, str, np.int64, np.int32], _weekday: int,
                                              _include: bool = False) -> datetime.datetime:
    return get_nearest_weekday_datetime_by_datetime(get_datetime_from_timestamp(_timestamp), _weekday, _include)


def get_nearest_weekday_datetime_by_date_str(_date_str: str, _weekday: int,
                                             _date_str_style: str = __DEFAULT_DATE_STR_STYLE,
                                             _include: bool = False) -> datetime.datetime:
    return get_nearest_weekday_datetime_by_datetime(datetime.datetime.strptime(_date_str, _date_str_style), _weekday,
                                                    _include)


# 将 datetime.date 或者 datetime.datetime 类型装华为指定样式的字符串
# 注意文件名里面不能有 [":"]
def change_datetime_to_str_by_style(_datetime: Union[datetime.datetime, datetime.date], _out_style: str) -> str:
    return _datetime.strftime(_out_style)


change_datetime_to_str = change_datetime_to_str_by_style


# 获得某一天是星期几，返回值为 range(1, 8)
def get_weekday_by_timestamp(_timestamp: Union[int, float, str, np.int64, np.int32], _pre_str: str = "星期") \
        -> Union[int, str]:
    weekday = get_weekday_by_datetime(get_datetime_from_timestamp(_timestamp))
    # python 中的 0 表示周日
    if weekday == 0:
        weekday = 7
    if _pre_str is not None:
        weekday = str(_pre_str) + str(weekday)
    return weekday


# 获得某一天是星期几，返回值为 range(1, 8)
def get_weekday_by_date_str(_date_str: str, _date_str_style=__DEFAULT_DATE_STR_STYLE, _pre_str="星期") \
        -> Union[int, str]:
    weekday = get_weekday_by_datetime(datetime.datetime.strptime(_date_str, _date_str_style))
    # python 中的 0 表示周日
    if weekday == 0:
        weekday = 7
    if _pre_str is not None:
        weekday = str(_pre_str) + str(weekday)
    return weekday


# 获得某个日期对应的月份
def get_month_by_date(_date_str: str, _date_str_style=__DEFAULT_DATE_STR_STYLE,
                      _month_str_style=__DEFAULT_MONTH_STR_STYLE) -> str:
    return datetime.datetime.strptime(_date_str, _date_str).strftime(_month_str_style)


# 给定其中日期和终止日期，获得月份
def get_month_list_by_start_and_end(_start_date: str, _end_date: str, _date_str_style=__DEFAULT_DATE_STR_STYLE,
                                    _month_str_style=__DEFAULT_MONTH_STR_STYLE) -> List[str]:
    date_list: List[str] = get_date_list_by_start_and_end(_start_date, _end_date, _date_str_style)
    return sorted({get_month_by_date(date, _date_str_style, _month_str_style) for date in date_list})


# 通过指定元祖创建日期
def create_date_by_tuple(_year: int, _month: int, _day: int) -> datetime.date:
    return datetime.date(_year, _month, _day)


# 通过指定元祖创建日期时间
def create_datetime_by_tuple(_year: int, _month: int, _day: int, _hour, _minute, _second) -> datetime.datetime:
    return datetime.datetime(_year, _month, _day, _hour, _minute, second_cn)


# 通过月份字符串获得该月所有的天
def get_date_list_by_month(_month_str: str, _month_str_style=__DEFAULT_MONTH_STR_STYLE,
                           _date_str_style=__DEFAULT_DATE_STR_STYLE) -> List[str]:
    month = datetime.datetime.strptime(_month_str, _month_str_style)
    year = month.year
    month = month.month
    days = calendar.monthrange(year, month)[1]
    start_date = change_datetime_to_str_by_style(create_date_by_tuple(year, month, 1), _date_str_style)
    end_date = change_datetime_to_str_by_style(create_date_by_tuple(year, month, days), _date_str_style)
    return get_date_list_by_start_and_end(start_date, end_date, _date_str_style)


# 获得两个天数的差，_include 表示是否 + 1
def get_date_diff(_start_date_str: str, _end_date_str: str, _date_str_style=__DEFAULT_DATE_STR_STYLE, _include=False,
                  _ignore_date_str_list=None) -> int:
    start_date = datetime.datetime.strptime(_start_date_str, _date_str_style)
    end_date = datetime.datetime.strptime(_end_date_str, _date_str_style)

    involve_count = 0
    involve_end_date_count = 0
    if _ignore_date_str_list is not None:
        for ignore_date_str in set(to_list(_ignore_date_str_list)):
            ignore_date = datetime.datetime.strptime(ignore_date_str, _date_str_style)
            if start_date <= ignore_date < end_date:
                involve_count += 1
            if ignore_date == end_date:
                involve_end_date_count = 1
    date_diff = (end_date - start_date).days - involve_count
    if _include:
        return date_diff + 1 - involve_end_date_count
    return date_diff


# 获得两个列表对应元素的天数的差
def get_date_list_diff(_start_date_str_list: List[str], _end_date_str_list: List[str],
                       _date_str_style=__DEFAULT_DATE_STR_STYLE) -> List[int]:
    assert_equal(len(_start_date_str_list), len(_end_date_str_list))
    return [get_date_diff(start_date_str, end_date_str, _date_str_style) for start_date_str, end_date_str in
            zip(_start_date_str_list, _end_date_str_list)]


# 从时间戳获得 datetime
def get_datetime_from_timestamp(_timestamp: Union[int, float, str, np.int64, np.int32]) -> datetime.datetime:
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
            raise ValueError(
                F"只支持 10(秒)、13(毫秒)、16(微妙)、19(纳秒)，不支持 {_timestamp}, length={len(str(_timestamp))}")
    else:
        raise ValueError(type(_timestamp))

    return datetime.datetime.fromtimestamp(_timestamp)


# 从时间戳获得时间字符串
def from_timestamp_to_time(_timestamp: Union[int, float, str, np.int64, np.int32]) -> str:
    date_time_str = get_datetime_from_timestamp(_timestamp).strftime("%Y/%m/%d %H:%M:%S")
    return date_time_str[11:19]


def get_current_datetime_str(_style: str = "%Y%m%d-%H%M%S") -> str:
    timestamp = get_current_timestamp()
    return change_datetime_to_str(get_datetime_from_timestamp(timestamp), _out_style=_style)


# 获得当前时间戳，支持 秒、毫秒、微妙、纳秒
# 注意 time.time 的小数点后只有 7 位
def get_current_timestamp(_digit=10) -> int:
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
def get_current_timestamp_second() -> int:
    return get_current_timestamp(10)


# 获得毫秒级时间戳
def get_current_timestamp_millisecond() -> int:
    return get_current_timestamp(13)


# 获得微秒级时间戳
def get_current_timestamp_microsecond() -> int:
    return get_current_timestamp(16)


# 获得纳秒级时间戳
def get_current_timestamp_nanosecond() -> int:
    return get_current_timestamp(19)


def main():
    date_str_style = "%Y%m%d"
    assert_equal(
        change_datetime_to_str_by_style(get_nearest_weekday_datetime_by_date_str("20200424", 5, date_str_style, False),
                                        date_str_style), "20200417")
    assert_equal(
        change_datetime_to_str_by_style(get_nearest_weekday_datetime_by_date_str("20200424", 5, date_str_style, True),
                                        date_str_style), "20200424")
    assert_equal(
        change_datetime_to_str_by_style(get_nearest_weekday_datetime_by_date_str("20200423", 5, date_str_style, False),
                                        date_str_style), "20200417")
    assert_equal(
        change_datetime_to_str_by_style(get_nearest_weekday_datetime_by_date_str("20200423", 5, date_str_style, True),
                                        date_str_style), "20200417")

    assert_equal(get_time_interval_str(0), "0 seconds")
    assert_equal(get_time_interval_str(-0), "0 seconds")
    assert_equal(get_time_interval_str(500), "8 minutes 20 seconds")
    assert_equal(get_time_interval_str(-500), "-8 minutes 20 seconds")
    assert_equal(get_time_interval_str(500.12345), "8 minutes 20 seconds 123 milliseconds")
    assert_equal(get_time_interval_str(-500.12345), "-8 minutes 20 seconds 123 milliseconds")
    assert_equal(get_time_interval_str(0.0004), "0 milliseconds")
    assert_equal(get_time_interval_str(-0.0004), "-0 milliseconds")

    # assert str(pd.to_datetime("2022-06-06 21-10", format="%Y-%m-%d")) == "2022-06-06 21:00:00-10:00"
    # from_datetime_str_to_datetime("2022-06-06 21-10","%Y-%m-%d")
    # assert (str(pd.to_datetime("2022-06-06 21:56:35.929", format="%Y-%m-%d")) ==
    #         "2022-06-06 21:56:35.929000")
    assert_equal(str(from_datetime_str_to_datetime("2022-06-06 21:56:35.929",
                                                   "%Y-%m-%d %H:%M:%S.%f")),
                 "2022-06-06 21:56:35.929000")
    assert not is_str_right_datetime_style("2022-06-06 21:56:35.929", "%Y-%m-%d")
    # assert str(pd.to_datetime(20220606, format="%Y-%m-%d")) == "1970-01-01 99:00:00.020220606"
    assert_equal(str(pd.to_datetime(20220606, format="%Y%m%d")), "2022-06-06 00:00:00")
    assert_equal(str(from_datetime_str_to_datetime("20220606", "%Y%m%d")), "2022-06-06 00:00:00")
    # from_datetime_str_to_datetime("20220606", "%Y-%m-%d")
    assert not is_str_right_datetime_style("20220606", "%Y-%m-%d")
    assert is_str_right_datetime_style("20220606", "%Y%m%d")
    assert not is_str_right_datetime_style("20241313", "%Y%m%d")

    assert_equal(get_date_diff("20220616", "20220727", date_str_style, True), 42)
    assert_equal(get_date_list_by_start_and_end("20220629", "20220702", date_str_style),
                 ["20220629", "20220630", "20220701", "20220702"])

    assert_equal(get_timezone(), -8)
    assert_equal(get_hour(from_datetime_str_to_datetime("2022-06-06 02:56:35.929", "%Y-%m-%d %H:%M:%S.%f")), 2)
    assert_equal(get_minute(from_datetime_str_to_datetime("2022-06-06 02:56:35.929", "%Y-%m-%d %H:%M:%S.%f")), 56)
    assert_equal(get_hour(from_datetime_str_to_datetime("23:56:35", "%H:%M:%S")), 23)
    assert_equal(get_minute(from_datetime_str_to_datetime("23:56", "%H:%M")), 56)

    assert_equal(get_date_with_delta_days("20230131", -1, "%Y%m%d"), "20230130")
    assert_equal(get_date_with_delta_days("20230201", -1, "%Y%m%d"), "20230131")
    assert_equal(get_date_with_delta_days("20230301", -1, "%Y%m%d"), "20230228")

    assert is_str_right_datetime_style("2023-06:29", "%Y-%m:%d")

    start_date = "20220616"
    end_date = "20220727"
    assert_equal(get_date_diff(start_date, end_date, date_str_style, True, ["20220601", "20220630", "20220730"]), 41)
    assert_equal(get_date_diff(start_date, end_date, date_str_style, False, ["20220601", "20220630", "20220730"]), 40)
    assert_equal(get_date_diff(start_date, end_date, date_str_style, True,
                               ["20220601", "20220630", "20220730", end_date, end_date]), 40)
    assert_equal(get_date_diff(start_date, end_date, date_str_style, False,
                               ["20220601", "20220630", "20220730", end_date, end_date]), 40)

    assert is_date_str_list_continuous(["20230728", "20230730", "20230729"], "%Y%m%d")
    assert not is_date_str_list_continuous(["20230728", "20230730", "20230729", "20230726"], "%Y%m%d")

    assert_equal(add_minute_on_minute_time_slice(202311121145, 16), 202311121201)
    assert_equal(add_minute_on_minute_time_slice(202311122345, 16), 202311130001)
    assert_equal(add_minute_on_minute_time_slice(202311302345, 16), 202312010001)

    assert_equal(add_minute_on_minute_time_slice(202311121145, 5), 202311121150)
    assert_equal(add_minute_on_minute_time_slice(202311121145, 15), 202311121200)
    assert_equal(add_minute_on_minute_time_slice(202311122345, 25), 202311130010)
    assert_equal(add_minute_on_minute_time_slice(202311122345, 35), 202311130020)

    assert_equal(get_two_minute_time_slice_gap(202311121145, 202311121150), 5)
    assert_equal(get_two_minute_time_slice_gap(202311121145, 202311121151), 6)
    assert_equal(get_two_minute_time_slice_gap(202311121145, 202311121200), 15)
    assert_equal(get_two_minute_time_slice_gap(202311121145, 202311121210), 25)
    assert_equal(get_two_minute_time_slice_gap(202311121155, 202311121220), 25)

    _datetime = pd.to_datetime("2023-11-13 22:30:41.739", format="%Y-%m-%d %H:%M:%S.%f")
    assert_equal(get_datetime_minute_time_slice(_datetime), 202311132230)
    assert_equal(get_datetime_minute_time_slice(_datetime, _extend_minutes_list=[35, 46]),
                 [202311132230, 202311132305, 202311132316])


if __name__ == '__main__':
    main()
