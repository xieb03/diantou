import requests
from util_date import *


# 获得get请求结果
def get_get_response_from_url(_url, _params=None, _timeout=120, _try_times=3, _print=False, **kwargs):
    if _print:
        print(_url)
    while _try_times >= 0:
        try:
            result = requests.get(url=_url, params=_params, timeout=_timeout, **kwargs)
            if result.status_code != 200:
                raise ValueError(F"{result.status_code}, 状态码错误.")
            return result
        except Exception as e:
            print(type(e))
            print(e)
            time.sleep(3)
            _try_times -= 1
            if _try_times > 0:
                print("最后第 {try_times} 次尝试".format(try_times=_try_times))
            else:
                raise e


def main():
    pass


if __name__ == '__main__':
    main()
