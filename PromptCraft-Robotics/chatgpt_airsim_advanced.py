from multiprocessing import Queue, Process

from airsim_wrapper import *
from rag_related import *
from recorder import Recorder

# 获取 rag 的 chromadb 的 collection
collection = ChromadbPersistentCollection(collection_name=CHROMADB_COLLECTION_NAME)

# 一些指令转移到历史记录里面，这样不用每次都提示一大堆
# prompt_template = ("You will be shown a new question, and some relevant historical dialogue records. "
#                    "You can refer to these records but should use the actual distance, direction, and coordinates "
#                    "from the new question. Specifically, do not use the location of related objects in historical "
#                    "records, but use code to find the actual location. "
#                    "You must explain how you use the records, however, if you feel that the relevant records "
#                    "are unreasonable or useless, you can ignore them, but must tell me the reasons."
#                    "\n\nnew question: {question}\n\n{count} relevant records: \n{records}")

prompt_template = "new question: {question}\n\n{count} relevant records: \n{records}"


# 开启 chatgpt_airsim
def start_chatgpt_airsim(_with_text=True, _queue: Queue = None):
    with open("prompt/system_prompt.txt", "r", encoding="UTF8") as f:
        system_prompt = f.read()

    with open("prompt/user_prompt.txt", "r", encoding="UTF8") as f:
        user_prompt = f.read()

    history_message_list = list()

    print(f"Initializing AirSim...")
    # noinspection PyUnusedLocal
    aw = AirSimWrapper()
    print("Done.")

    response = get_chat_completion_content(user_prompt=user_prompt, system_prompt=system_prompt,
                                           history_message_list=history_message_list)
    # Understood. I am ready for the new question and will refer to the historical dialogue records as needed.
    print(Colors.BLUE + f"\n{response}\n" + Colors.ENDC)

    # print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")

    while True:
        if _with_text:
            question = input(Colors.YELLOW + "AirSim> " + Colors.ENDC)
            if question == "!quit" or question == "!exit":
                break

            if question == "!clear":
                os.system("cls")
                continue
        else:
            print(Colors.YELLOW + "请说出你的指令> ")
            # 利用 block=True，设置等待
            question = _queue.get(block=True)
            print(Colors.YELLOW + F"你的指令是 '{question}'" + Colors.ENDC)

            if question.startswith("退出"):
                break

            if question.startswith("清空屏幕"):
                os.system("cls")
                continue

        rag_question_list, rag_answer_list = get_rag_results(_question=question, _collection=collection,
                                                             _debug=True, _top_n=3, _min_similarity=0.4)
        prompt = assemble_prompt_from_template(question, rag_question_list, rag_answer_list, prompt_template)
        print(Colors.GREEN + F"你的最终 prompt 是：\n{prompt}" + Colors.ENDC)

        response = get_chat_completion_content(user_prompt=prompt, history_message_list=history_message_list)

        print(Colors.BLUE + f"\n{response}\n" + Colors.ENDC)

        code = extract_python_code(response)
        if code is not None:
            if not check_python_code_syntax_error(code):
                print(Colors.RED + f"获取的 python 代码编译有问题，请重新试过." + Colors.ENDC)
                continue
            else:
                print("Please wait while I run the code in AirSim...")
                exec(code)
                print("Done!\n")
        else:
            print("Sorry, i didn't get any code from the response, please try again.")

    sys.exit(0)


# 开启 recorder
def start_recoder(_model_name="medium", _queue: Queue = None):
    recorder = Recorder(_model_name=_model_name, _queue=_queue)
    recorder.run()


def main():
    with_text = True
    # with_text = False

    if with_text:
        start_chatgpt_airsim(True)
    else:
        # chatgpt 和 recoder 通信，借助一个共享队列来传递文本
        shared_queue = Queue()

        # text -> chatgpt response -> code -> airsim 模拟
        p1 = Process(target=start_chatgpt_airsim, args=(False, shared_queue))
        p1.start()

        # tkinter 界面 -> sounddevice 录音 -> mp3 -> local whisper 语音识别 -> text
        p2 = Process(target=start_recoder, args=("medium", shared_queue))
        p2.start()

        p1.join()
        p2.join()


if __name__ == '__main__':
    main()
