import sys
import threading
import tkinter
from multiprocessing import Queue

sys.path.append("../")
from project_utils import *


class Recorder:
    def __init__(self, _model_name="medium", _queue: Queue = None):
        # signal 用来进行开始录音和结束录音的标记位
        self.signal = False
        # whisper 本地模型，预先加载好
        self.model = get_whisper_model_local(_model_name=_model_name)
        self.tk = tkinter.Tk()

        screen_width = self.tk.winfo_screenwidth()
        screen_height = self.tk.winfo_screenheight()
        x = int(screen_width * 0.5)
        y = int(screen_height * 0.33)
        self.tk.title("Voice Recorder")
        self.tk.geometry(F"400x150+{x}+{y}")
        self.tk.resizable(True, True)
        self.label = tkinter.Label(self.tk, text="准备开始", width=x // 2)
        self.label.pack()
        self.start_button = tkinter.Button(self.tk, text="开始录音", pady=0, state="normal", command=self.start_record)
        self.start_button.place(relx=0.3, rely=0.25, relwidth=0.4, relheight=0.3)
        self.stop_button = tkinter.Button(self.tk, text="结束录音", pady=0, state="disabled", command=self.stop_record)
        self.stop_button.place(relx=0.3, rely=0.6, relwidth=0.4, relheight=0.3)

        self.file_path = None
        self.queue = _queue

    def run(self):
        self.tk.mainloop()

    def start_record(self):
        self.signal = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.label.config(text="正在录音中...")

        # 创建一个新线程，以免在主线程中进行这些操作时阻塞了Tkinter主循环，导致界面无法响应
        t = threading.Thread(target=self._start_record)
        # 启动线程
        t.start()

    # 停止录音后
    def _after_record(self):
        self.label.config(text="录音结束，开始进行语音识别：")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")

        # 为了给中文增加标点符号，同时尽量用普通话进行转译
        text = get_whisper_text_local(self.file_path, _model=self.model,
                                      _initial_prompt="以下是普通话的句子，这是一个指令。")
        self.label.config(text=f"识别结果为：{text}")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        if self.queue is not None:
            self.queue.put(text)

    def _start_record(self):
        self.file_path = BIGDATA_VOICES_PATH + get_current_datetime_str() + ".mp3"
        q = queue.Queue()

        # noinspection PyUnusedLocal
        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        # noinspection PyBroadException
        try:
            with sf.SoundFile(self.file_path, mode='x', samplerate=44100,
                              channels=1, subtype="MPEG_LAYER_III") as file:
                with sd.InputStream(samplerate=44100, device=get_available_microphone_by_name()["index"],
                                    channels=1, callback=callback):
                    while True:
                        if not self.signal:
                            self._after_record()
                            break
                        file.write(q.get())
        except Exception:
            traceback.print_exc()

    def stop_record(self):
        self.signal = False
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")


def main():
    recorder = Recorder("base")
    recorder.run()


if __name__ == '__main__':
    main()
