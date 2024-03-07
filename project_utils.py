import sys

import pydub
import sounddevice as sd
import soundfile as sf

# noinspection PyUnresolvedReferences
from util_chatglm import *
# noinspection PyUnresolvedReferences
from util_chromadb import *
from util_openai import *
from util_path import *


# 'check_edge_tts' spent 12.0146s.
@func_timer()
def check_edge_tts():
    text = "我是谢博，我爱老婆平平！"
    # 去掉所有空格
    text = delete_all_blank(text)
    file_path = BIGDATA_VOICES_PATH + text[:20] + ".mp3"
    voice = "zh-CN-YunyangNeural"
    cmd = F"edge-tts --text '{text}' --write-media {file_path} --rate +0% --voice {voice} --volume +50%"
    os.system(cmd)


# 不可用
# def print_all_available_microphones(_p=None, _print=True):
#     need_close = False
#     if _p is None:
#         _p = pyaudio.PyAudio()
#         need_close = True
#
#     host_api_filter = pyaudio.paDirectSound
#
#     device_info_list = list()
#     for i in range(_p.get_device_count()):
#         device_info = _p.get_device_info_by_index(i)
#         if not device_info.get("maxInputChannels", 0) > 0:
#             continue
#         if device_info.get("hostApi") != host_api_filter:
#             continue
#         device_info_list.append(device_info)
#     if need_close:
#         _p.terminate()
#
#     if _print:
#         print_list(device_info_list, _start_index=1)
#
#     return device_info_list


# 不可用
# def get_available_microphone_by_name(_p=None, _name="麦克风 (Logitech BRIO)"):
#     device_info_list = print_all_available_microphones(_p, _print=False)
#     name_list = list()
#     for device_info in device_info_list:
#         if device_info["name"].startswith(_name):
#             return device_info
#         else:
#             name_list.append(device_info["name"])
#
#     raise ValueError(F"没有找到形如 {_name} 的麦克风，目前系统中可用的麦克风有：{name_list}.")

def get_available_microphone_by_name(_name="麦克风 (Logitech BRIO)"):
    device_list = sd.query_devices(kind="input")
    if isinstance(device_list, dict):
        if device_list["name"] != _name:
            raise ValueError(F"没有找到形如 {_name} 的麦克风，目前系统中可用的麦克风只有：{device_list['name']}.")
        else:
            return device_list
    else:
        name_list = list()
        for device in device_list:
            if device["name"] == _name:
                return device
            else:
                name_list.append(device["name"])

        raise ValueError(F"没有找到形如 {_name} 的麦克风，目前系统中可用的麦克风有：{name_list}.")


# 录制一段固定时长的语音输入
# https://www.pythonheidong.com/blog/article/2000048/e6b613a3c2f21a85837e/
def record_voice_fix_duration(_duration, _file_path=None, _channels=1):
    # 位深
    bit = 'int16'
    # 采样频率，每秒采多少个点
    fs = 44100

    print("开始录音")
    data: np.ndarray = sd.rec(frames=int(_duration * fs), samplerate=fs, channels=_channels, dtype=bit,
                              device=get_available_microphone_by_name()["index"], blocking=True)
    print("结束录音")

    # sample_width = 2 对应 bit = int16
    voice = pydub.AudioSegment(data.tobytes(), frame_rate=fs, sample_width=2, channels=_channels)
    if _file_path is None:
        _file_path = BIGDATA_VOICES_PATH + get_current_datetime_str() + ".mp3"
    voice.export(_file_path, format="mp3", bitrate=None)


# 录制一段任意时长的语音输入，Ctrl + C 结束
# https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html#recording-with-arbitrary-duration
# 'record_voice_arbitrary_duration' spent 15.3748s.
@func_timer()
def record_voice_arbitrary_duration(_file_path=None, _channels=1, _with_wisper=True):
    if _file_path is None:
        _file_path = BIGDATA_VOICES_PATH + get_current_datetime_str() + ".mp3"
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # 采样频率，每秒采多少个点
    fs = 44100

    # noinspection PyBroadException
    try:
        with sf.SoundFile(_file_path, mode='x', samplerate=fs,
                          channels=_channels, subtype="MPEG_LAYER_III") as file:
            with sd.InputStream(samplerate=fs, device=get_available_microphone_by_name()["index"],
                                channels=_channels, callback=callback):
                print('#' * 80)
                print("开始录音，按 Ctrl + C 停止录音.")
                print('#' * 80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        if not _with_wisper:
            print(F"录音结束，文件保存到：{_file_path}")
        else:
            print("录音结束，开始进行语音识别：")
            # 为了给中文增加标点符号，同时尽量用普通话尽心转译
            text = get_whisper_text_local(_file_path, _model_name="medium",
                                          _initial_prompt="以下是普通话的句子，这是一个指令。")
            print(f"识别结果为：{text}")
    except Exception:
        traceback.print_exc()


# 录制一段固定时长的语音输入
# https://www.pythonheidong.com/blog/article/2000048/e6b613a3c2f21a85837e/
# def record_voice_fix_duration_using_pyaudio(_file_path=None):
#     import pyaudio
#     import wave
#     p = pyaudio.PyAudio()
#     input_device_index = int(get_available_microphone_by_name(p)["index"])
#
#     # 位深，表示对于响度的精确控制，格式为16位有符号整型
#     bit = pyaudio.paInt16
#     sample_size = p.get_sample_size(bit)
#     # 采样频率，每秒采多少个点
#     fs = 44100
#     # 2，双通道（立体声）
#     channels = 2
#     # 处理并行度
#     chunk = 4096
#     # 每次读取的音频帧大小
#     record_second = 5
#     num_times = record_second * fs // chunk
#     print(num_times)
#
#     # rate: Sampling rate
#     # channels: Number of channels
#     # Sampling size and format. See |PaSampleFormat|.
#     # Specifies whether this is an input stream. Defaults to ``False``.
#     stream = p.open(format=bit, channels=channels, rate=fs, input=True, output=False, frames_per_buffer=chunk,
#                     input_device_index=input_device_index)
#     stream.start_stream()
#     print("录制开始")
#     frames = list()
#     for i in range(num_times):
#         # b'\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\xff\x01'
#         data = stream.read(chunk)
#         print(data)
#         frames.append(data)
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     print("结束")
#
#     file_path = BIGDATA_VOICES_PATH + get_current_datetime_str() + ".wav"
#     with wave.open(file_path, 'wb') as wf:
#         wf.setnchannels(channels)
#         wf.setsampwidth(sample_size)
#         wf.setframerate(fs)
#         wf.writeframes(b"".join(frames))

def main():
    check_edge_tts()
    record_voice_arbitrary_duration()


if __name__ == '__main__':
    main()
