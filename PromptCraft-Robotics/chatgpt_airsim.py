import sys

from airsim_wrapper import *

sys.path.append("../")
from project_utils import *

with open("system_prompt.txt", "r") as f:
    system_prompt = f.read()

with open("user_prompt.txt", "r") as f:
    user_prompt = f.read()

history_message_list = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": "move 10 units up"
    },
    {
        "role": "assistant",
        "content": """```python
aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])
```

This code uses the `fly_to()` function to move the drone to a new position that is 10 units up from the current position. It does this by getting the current position of the drone using `get_drone_position()` and then creating a new list with the same X and Y coordinates, but with the Z coordinate increased by 10. The drone will then fly to this new position using `fly_to()`."""
    }
]

print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


print(f"Initializing AirSim...")
aw = AirSimWrapper()
print(f"Done.")

response = get_chat_completion_content(user_prompt=user_prompt, system_prompt=system_prompt,
                                       history_message_list=history_message_list)
# Understood. Thank you for providing the functions and clarifications.
# Please let me know what you would like me to do with the drone.
print(f"\n{response}\n")

print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")

while True:
    question = input(colors.YELLOW + "AirSim> " + colors.ENDC)

    if question == "!quit" or question == "!exit":
        break

    if question == "!clear":
        os.system("cls")
        continue

    response = get_chat_completion_content(user_prompt=question, history_message_list=history_message_list)

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in AirSim...")
        exec(extract_python_code(response))
        print("Done!\n")
