import os
import sys
import fcntl
import termios
import struct
import argparse
import json
from io import StringIO
from abc import ABC, abstractmethod
import getpass
from threading import Thread, Event

from openai import OpenAI
import pyte
import pexpect


class DSChat(ABC):

    @abstractmethod
    def __init__(self, sys_prompt, client, model, temperature) -> None:
        pass

    @abstractmethod
    def __call__(self, content) -> str:
        pass


class StatelessChat(DSChat):

    def __init__(self, sys_prompt, client, model, temperature=0.3):
        self.sys_prompt = sys_prompt
        self.model = model
        self.client = client
        self.temperature = temperature

    def __call__(self, content) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": self.sys_prompt
            }, {
                "role": "user",
                "content": content,
            }],
            temperature=self.temperature,
            stream=True,
        )

        result = ""

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if (not hasattr(delta, 'reasoning_content')
                    ) or delta.reasoning_content is None:
                    result += delta.content

        return result


class ChatBot(DSChat):

    def __init__(self, sys_prompt, client, model, temperature=0.3):
        self.model = model
        self.messages = [{"role": "system", "content": sys_prompt}]
        self.client = client
        self.temperature = temperature

    def __call__(self, content) -> str:
        self.messages.append({
            "role": "user",
            "content": content,
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            stream=True)

        result = ""

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if (not hasattr(delta, 'reasoning_content')
                    ) or delta.reasoning_content is None:
                    result += delta.content

        self.messages.append({"role": "assistant", "content": result})
        return result

    def retry(self):
        assert self.messages[-1]["role"] == "assistant"
        assert self.messages[-2]["role"] == "user"
        self.messages.pop()
        user = self.messages.pop()["content"]
        return self(user)

    def clear_ctx(self):
        self.messages = [self.messages[0]]


def main():
    parser = argparse.ArgumentParser(description="处理LLM相关参数")
    parser.add_argument('--prompt', type=str, help='用户的需求描述')
    parser.add_argument('--llm-url',
                        default="https://api.deepseek.com",
                        type=str,
                        help='LLM模型的API服务地址')
    parser.add_argument('--llm-model',
                        default="deepseek-chat",
                        type=str,
                        help='要使用的LLM模型名称')
    parser.add_argument('--llm-key',
                        required=True,
                        type=str,
                        help='访问LLM API的认证密钥')
    parser.add_argument('--no-check', action='store_true', help='是否禁用检查（开关参数）')
    args = parser.parse_args()

    client = OpenAI(api_key=args.llm_key, base_url=args.llm_url)

    system_prompt = """\
你是一个擅长执行类unix终端命令的专家，你可以控制 user 的终端。\
你的工作是根据 user 的诉求和已经执行的命令，\
决定下一步需要在终端执行的命令。
输出格式为 JSON:
{ \"finished\": true|false, \"idea\": \"你的思路\", \"cmd\": \"ls -l xxx\"}
当你认为任务已经完成的时候，请将 finished 设置为 true, 并在 idea 中向用户汇报任务结果；\
如果你觉得任务还要继续，请在 idea 当中简单说一下你下一步的思路，并在 cmd 中给出下一步的命令。
注意：
1. 在任何情况下请不要输出任何其它内容
2. 一次请只执行一条命令
3. 可以执行 cd 命令
"""

    bot = ChatBot(system_prompt, client=client, model=args.llm_model)
    if args.prompt:
        user_prompt = args.prompt
    else:
        user_prompt = agent_input("Your prompt > ")
    bot_words = bot(user_prompt)
    runner = Runner()
    user_name = getpass.getuser()
    first = True

    while True:
        try:
            bot_words = json.loads(bot_words)
        except json.JSONDecodeError:
            agent_print("llm output parsing failed:", bot_words, "retry...")
            bot_words = bot.retry()
            continue
        if bot_words["finished"] and not first:
            break
        first = False
        agent_print(bot_words["idea"])
        cmd: str = bot_words["cmd"]
        agent_print_cmd(cmd)
        if (not args.no_check) and (user := agent_input(
                "run(enter) or reject and say to ai?").strip()) != "":
            ret = {
                "user": user_name,
                "cwd": runner.cwd(),
                "return_code": 255,
                "output": f"User rejected your command. User said: {user}"
            }
        else:
            ret_code, output = runner(cmd)
            ret = {
                "user": user_name,
                "cwd": runner.cwd(),
                "return_code": ret_code,
                "output": output,
            }
        bot_words = bot(str(ret))

    summary_print(bot.messages)
    agent_print(bot_words["idea"])


class Runner:

    def __init__(self):
        self.proc = pexpect.spawn("sh",
                                  echo=True,
                                  env=os.environ.update({"TERM": "linux"}))
        self.__update_win_size()
        # end_str 不要直接写出来，不然 cat main.py 会出问题
        self.end_str = "\x5f\x5f\x41\x55\x54\x4f\x52\x4d\x49\x4e\x41\x4c\x5f\x45\x4e\x44\x5f\x5f"
        self.stop = Event()
        self.proc.send(f"export PS1={self.end_str}\n")

    def __update_win_size(self):
        try:
            fd = sys.stdout.fileno()
            hw = struct.unpack(
                'hhhh',
                fcntl.ioctl(fd, termios.TIOCGWINSZ,
                            struct.pack('hhhh', 0, 0, 0, 0)))
            self.proc.setwinsize(hw[0], hw[1])
        except Exception:
            pass

    def __copy_stdin(self):
        while not self.stop.is_set():
            ch = sys.stdin.read(1)
            if ch != '':
                self.proc.send(ch)

    def __read_output_all(self) -> bytes:
        result = b''
        while True:
            try:
                data = self.proc.read_nonblocking(1024, timeout=0.1)
            except Exception:
                break
            if data:
                result += data
        return result

    def __copy_output(self, echo=True) -> bytes:
        output = b''
        while True:
            data = self.__read_output_all()
            if len(data) == 0:
                continue
            if bytes(self.end_str, encoding="utf-8") not in data:
                if echo:
                    try:
                        sys.stdout.buffer.write(data)
                        sys.stdout.flush()
                    except BlockingIOError:
                        pass
                output += data
            else:
                break

        data, _ = data.split(bytes(self.end_str, encoding="utf-8"), 1)
        if echo:
            try:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
            except BlockingIOError:
                pass
        output += data

        return output

    def __get_final_text(self, input_bytes):
        screen = pyte.Screen(500, 3000)
        stream = pyte.ByteStream(screen)
        stream.feed(input_bytes)
        final_lines = [line.rstrip() for line in screen.display]
        return '\n'.join(final_lines).rstrip('\n')

    def __setup_terminal(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] &= ~termios.ICANON
        new_settings[3] &= ~termios.ECHO
        os.set_blocking(fd, False)
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        return old_settings

    def __restore_terminal(self, settings):
        fd = sys.stdin.fileno()
        os.set_blocking(fd, True)
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)

    def cwd(self) -> str:
        _, path = self("pwd", _echo=False, _ret_code=False)
        return path

    def __last_ret_code(self) -> int:
        _, code = self("echo $?", _echo=False, _ret_code=False)
        return int(code)

    def __call__(self, cmd: str, _echo=True, _ret_code=True) -> (int, str):
        self.__update_win_size()
        self.__read_output_all()

        cmd = bytes(cmd + '\n', encoding="utf-8")
        self.proc.send(cmd)
        self.proc.read_nonblocking(len(cmd) + 1, timeout=1)
        settings = self.__setup_terminal()

        self.stop.clear()
        thread = Thread(target=self.__copy_stdin)
        thread.daemon = True
        thread.start()
        output = self.__copy_output(_echo)
        self.stop.set()
        self.__restore_terminal(settings)
        output = self.__get_final_text(output)
        thread.join()

        if _ret_code:
            code = self.__last_ret_code()
        else:
            code = None

        return code, output


def agent_print(*args, **kwargs):
    print("\033[94m", end="")
    sio = StringIO()
    print(*args, **kwargs, file=sio)
    sio.seek(0)
    for line in sio.readlines():
        print("[Autorminal]", line, end="")
    print("\033[0m", end="", flush=True)


def agent_print_cmd(cmd: str):
    print(f"\033[94m[Autorminal]\033[0m \033[93m{cmd}\033[0m", flush=True)


def agent_input(*args, **kwargs):
    print("\033[94m[Autorminal]\033[0m ", end="")
    print("\033[94m", end="")
    result = input(*args, **kwargs)
    print("\033[0m", end="", flush=True)
    return result


def summary_print(messages: list):
    agent_print("SUMMARY")
    agent_print(messages[1]["content"])
    for message in messages:
        if message["role"] == "assistant":
            cmd = json.loads(message["content"])["cmd"].strip()
            if cmd != "":
                agent_print_cmd(cmd)


if __name__ == "__main__":
    main()
