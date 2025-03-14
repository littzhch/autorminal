import os
import sys
import fcntl
import termios
import struct
import signal
import argparse
import select
import json
import time
import readline
from io import StringIO
from abc import ABC, abstractmethod
from contextlib import contextmanager
import getpass
from threading import Thread, Event
import subprocess
import pty

from openai import OpenAI
import pyte
import psutil


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

    def replace_last_reply(self, new_content):
        assert self.messages[-1]["role"] == "assistant"
        self.messages[-1]["content"] = new_content

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

    system_prompt = """\
你是一个擅长执行类unix终端命令的专家，你可以控制 user 的终端。\
你的工作是根据 user 提出的 task 和已经执行的命令\
决定下一步需要在终端执行的命令。
输出格式为 JSON:
{ \"idea\": \"你的思路\", \"cmd\": \"ls -l xxx\"}
当你认为任务已经完成的时候，请在 idea 中向用户汇报任务结果，并将 cmd 设置为空；\
如果你觉得任务还要继续，请在 idea 当中简单说一下你下一步的思路，并在 cmd 中给出下一步的命令。
注意：
1. 输出以 { 开始， } 结束，在任何情况下请不要输出任何其它内容
2. 一次请只执行一条命令
3. 可以执行 cd 命令
"""
    client = OpenAI(api_key=args.llm_key, base_url=args.llm_url)
    bot = ChatBot(system_prompt, client=client, model=args.llm_model)
    prompt_runner = PromptRunner(bot, check=not args.no_check)

    while True:
        prompt = read_prompt(prompt_runner.cwd())
        try:
            prompt_runner(prompt)
        except KeyboardInterrupt:
            pass


def read_prompt(cwd) -> str:
    agent_print(cwd)
    while True:
        try:
            prompt = agent_input("> ").strip()
            if prompt != "":
                return prompt
        except KeyboardInterrupt:
            print()
        except EOFError:
            print("\033[0m")
            exit()


def history_based_completer(text, state):
    matches = []
    history_length = readline.get_current_history_length()
    for i in range(1, history_length + 1):
        item = readline.get_history_item(i)
        if item.startswith(text):
            matches.append(item)

    unique_matches = []
    seen = set()
    for item in matches:
        if item not in seen:
            seen.add(item)
            unique_matches.append(item)

    return unique_matches[state] if state < len(unique_matches) else None


readline.set_completer(history_based_completer)
readline.parse_and_bind("tab: complete")


class PromptRunner:

    def __init__(self, chatbot: ChatBot, check=True):
        self.check = check
        self.bot = chatbot
        self.runner = Runner()
        self.user_name = getpass.getuser()

    def cwd(self):
        return self.runner.cwd()

    def __call__(self, user_prompt):
        self.bot.clear_ctx()
        user_message = {
            "user": self.user_name,
            "cwd": self.cwd(),
            "task": user_prompt,
        }
        with spinning(" Thinking...", ["-", "\\", "|", "/"]):
            bot_words = self.bot(json.dumps(user_message, ensure_ascii=False))

        while True:
            try:
                bot_words = json.loads(bot_words)
            except json.JSONDecodeError:
                agent_print("llm output parsing failed:", bot_words,
                            "retry...")
                bot_words = self.bot.retry()
                continue

            if bot_words["cmd"] == "":
                break

            agent_print(bot_words["idea"])
            cmd: str = bot_words["cmd"]
            agent_print_cmd(cmd)
            if self.check and (user := agent_input(
                    "run(enter) or reject and say to ai > ").strip()) != "":
                ret = {
                    "user": self.user_name,
                    "cwd": self.runner.cwd(),
                    "return_code": 255,
                    "output": f"User rejected your command. User said: {user}"
                }
            else:
                ret_code, output = self.runner(cmd)
                if ret_code == 0:
                    agent_print("\033[32m0 OK\033[0m")
                else:
                    agent_print(f"\033[31m{ret_code} ERR\033[0m")
                ret = {
                    "user": self.user_name,
                    "cwd": self.runner.cwd(),
                    "return_code": ret_code,
                    "output": output,
                }
            with spinning(" Thinking...", ["-", "\\", "|", "/"]):
                bot_words = self.bot(str(ret))

        summary_print(self.bot.messages)
        agent_print(bot_words["idea"])
        print()


class Runner:

    def __init__(self):
        self.proc = Terminal(["bash", "--posix"])
        self.__update_win_size()
        # end_str 不要直接写出来，不然 cat main.py 会出问题
        self.end_str = b"\x5f\x5f\x41\x55\x54\x4f\x52\x4d\x49\x4e\x41\x4c\x5f\x45\x4e\x44\x5f\x5f"

        self.copy_input_run = Event()
        self.copy_input_thread = Thread(target=self.__copy_input)
        self.copy_input_thread.daemon = True
        self.copy_input_thread.start()
        self.proc.send(f"PROMPT_COMMAND=\"PS1={self.end_str.decode()}\"\n")
        self.__run_simple("readonly PROMPT_COMMAND")

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

    def __copy_input(self):
        while self.copy_input_run.wait():
            readable, _, _ = select.select([sys.stdin], [], [])
            if not self.copy_input_run.is_set():
                continue
            if sys.stdin not in readable:
                continue
            ch = sys.stdin.buffer.read(4096)
            if ch is not None:
                self.proc.send(ch)

    def __clean_output(self):
        while True:
            try:
                self.proc.read_nonblocking(4096)
            except BlockingIOError:
                break

    def __copy_output(self) -> bytes:
        output = b''
        stdout_ptr = 0
        end_str_len = len(self.end_str)
        while self.end_str not in output:
            try:
                if len(output) - stdout_ptr > end_str_len:
                    sys.stdout.buffer.write(output[stdout_ptr:-end_str_len])
                    stdout_ptr = len(output) - end_str_len
                for _ in range(0, len(output) - stdout_ptr):
                    if not self.end_str.startswith(output[stdout_ptr:]):
                        sys.stdout.buffer.write(output[stdout_ptr:stdout_ptr +
                                                       1])
                        stdout_ptr += 1
                    else:
                        break
                sys.stdout.flush()
            except BlockingIOError:
                pass

            output += self.proc.read(4096)

        while True:
            try:
                sys.stdout.buffer.write(output[stdout_ptr:-end_str_len])
                break
            except BlockingIOError:
                pass

        while True:
            try:
                sys.stdout.flush()
                break
            except BlockingIOError:
                pass

        output = output.split(self.end_str, 1)[0]
        return output

    def __get_final_text(self, input_bytes):
        screen = pyte.Screen(500, 3000)
        stream = pyte.ByteStream(screen)
        stream.feed(input_bytes)
        final_lines = [line.rstrip() for line in screen.display]
        return '\n'.join(final_lines).rstrip('\n')

    def cwd(self) -> str:
        path = self.__run_simple("pwd")
        return path

    def __last_ret_code(self) -> int:
        code = self.__run_simple("echo $?")
        return int(code)

    def __run_simple(self, cmd) -> str:
        self.__clean_output()
        cmd = bytes(cmd + '\n', encoding="utf-8")
        self.proc.send(cmd)
        self.proc.read_exact(len(cmd) + 1)
        result = b''
        while self.end_str not in result:
            result += self.proc.read(1024)
        result = self.__get_final_text(result.split(self.end_str)[0])
        return result

    def __call__(self, cmd: str) -> tuple[int, str]:
        self.__update_win_size()
        self.__clean_output()

        cmd = bytes(cmd + '\n', encoding="utf-8")
        self.proc.send(cmd)
        self.proc.read_exact(len(cmd) + 1)

        with self.__running_context():
            output = self.__copy_output()

        output = self.__get_final_text(output)
        code = self.__last_ret_code()

        return code, output

    @contextmanager
    def __running_context(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] &= ~termios.ICANON
        new_settings[3] &= ~termios.ECHO
        os.set_blocking(fd, False)
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        self.copy_input_run.set()
        old_handler = signal.signal(signal.SIGINT,
                                    lambda signum, frame: self.proc.ctrlc())

        yield None

        signal.signal(signal.SIGINT, old_handler)
        self.copy_input_run.clear()
        fd = sys.stdin.fileno()
        os.set_blocking(fd, True)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class Terminal:

    def __init__(self, cmd):
        self.master_fd, self.slave_fd = pty.openpty()
        self.subp = subprocess.Popen(
            cmd,
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            start_new_session=True,
        )
        os.set_blocking(self.master_fd, False)

    def send(self, data):
        if isinstance(data, str):
            data = bytes(data, encoding="utf-8")
        count = len(data)
        while count > 0:
            count -= os.write(self.master_fd, data)

    def read_nonblocking(self, count):
        """read at most `count`` bytes"""
        return os.read(self.master_fd, count)

    def read(self, count):
        """read at most `count` bytes, blocking"""
        readable, _, _ = select.select([self.master_fd], [], [])
        assert self.master_fd in readable
        return os.read(self.master_fd, count)

    def read_exact(self, count):
        """read exactly count bytes, blocking"""
        data = b''
        while count > 0:
            try:
                d = os.read(self.master_fd, count)
            except BlockingIOError:
                d = b''
            count -= len(d)
            data += d

    def setwinsize(self, rows, cols):
        s = struct.pack('HHHH', rows, cols, 0, 0)
        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, s)

    def ctrlc(self):
        for p in psutil.Process(self.subp.pid).children(recursive=True):
            try:
                os.kill(p.pid, signal.SIGINT)
            except PermissionError:
                print("permission")


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
    args = list(args)
    args[0] = f"\033[94m[Autorminal] {args[0]}"
    result = input(*args, **kwargs)
    print("\033[0m", end="", flush=True)
    return result


def summary_print(messages: list):
    agent_print("SUMMARY")
    agent_print(json.loads(messages[1]["content"])["task"])
    for message in messages:
        if message["role"] == "assistant":
            cmd = json.loads(message["content"])["cmd"].strip()
            if cmd != "":
                agent_print_cmd(cmd)


@contextmanager
def spinning(text: str, spinner_seq: [str]):
    stop = Event()

    def spin():
        while True:
            for spinner in spinner_seq:
                print(f"{spinner}{text}\r", end="")
                time.sleep(0.3)
                if stop.is_set():
                    print(f"{' ' * (len(spinner) + len(text))}\r", end="")
                    return

    thread = Thread(target=spin)
    thread.daemon = True
    thread.start()
    print("\033[?25l", end="")

    try:
        yield None
    finally:
        print("\033[?25h", end="")
        stop.set()
        thread.join()


if __name__ == "__main__":
    main()
