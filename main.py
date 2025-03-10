import os
import sys
import pty
import fcntl
import termios
import struct
import argparse
import subprocess
import json
from io import StringIO
from abc import ABC, abstractmethod
import getpass
from threading import Thread

from openai import OpenAI
import pyte


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
            stream=False,
        )
        return response.choices[0].message.content


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
            stream=False)
        result = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": result})
        return result

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
        bot_words = args.prompt
    else:
        bot_words = bot(agent_input("Your prompt > "))
    runner = Runner()
    user_name = getpass.getuser()
    first = True

    while True:
        bot_words = json.loads(bot_words)
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
                "cwd": runner.cwd,
                "return_code": 255,
                "stdout": "",
                "stderr": f"User rejected your command. User said: {user}"
            }
        else:
            ret_code, stdout, stderr = runner(cmd)
            ret = {
                "user": user_name,
                "cwd": runner.cwd,
                "return_code": ret_code,
                "stdout": stdout,
                "stderr": stderr
            }
        bot_words = bot(str(ret))

    summary_print(bot.messages)
    agent_print(bot_words["idea"])


# def main():
#     chatbot = ChatBot(sys_prompt="你是一只可爱的猫娘呀～每句话都要以喵结尾～")
#     while True:
#         agent_say(chatbot(input('>')))


def run_command(cmd: str, cwd: str, extra_env=None) -> (int, str, str):

    def get_terminal_size():
        """获取当前终端的行数和列数"""
        try:
            # 通过标准输出的文件描述符获取窗口尺寸
            fd = sys.stdout.fileno()
            hw = struct.unpack(
                'hhhh',
                fcntl.ioctl(fd, termios.TIOCGWINSZ,
                            struct.pack('hhhh', 0, 0, 0, 0)))
            return (hw[0], hw[1])
        except Exception:
            return (24, 80)  # 默认值（行数24，列数80）

    def set_pty_size(fd):
        """设置伪终端的窗口尺寸"""
        rows, cols = get_terminal_size()
        # 构造TIOCSWINSZ命令所需的结构体
        size = struct.pack('hhhh', rows, cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, size)

    def read_stdout(fd, output):
        """从伪终端读取stdout并实时显示"""
        while True:
            try:
                data = os.read(fd, 1024)
            except OSError:  # 当伪终端关闭时退出
                break
            if not data:
                break
            decoded = data.decode(errors='replace')
            output.append(decoded)
            sys.stdout.buffer.write(data)
            sys.stdout.flush()

    def read_stderr(pipe, output):
        """从管道读取stderr并实时显示"""
        while True:
            data = os.read(pipe.fileno(), 1024)
            if not data:
                break
            decoded = data.decode(errors='replace')
            output.append(decoded)
            sys.stderr.write(decoded)
            sys.stderr.flush()

    # 获取当前终端尺寸
    rows, cols = get_terminal_size()

    # 创建伪终端（主端和从端）
    master_fd, slave_fd = pty.openpty()

    # 设置伪终端的窗口尺寸
    set_pty_size(master_fd)

    final_env = dict(os.environ, TERM=os.environ.get('TERM', 'xterm'))
    if extra_env:
        final_env.update(extra_env)

    # 启动子进程，将stdout和stdin绑定到伪终端，stderr使用管道
    proc = subprocess.Popen(
        [cmd],  # 替换为实际命令
        cwd=cwd,
        stdin=None,
        stdout=slave_fd,
        stderr=subprocess.PIPE,  # stderr通过普通管道传输
        close_fds=True,
        env=final_env,  # 传递终端类型
        #preexec_fn=os.setsid,  # 创建新进程组以正确处理信号
        shell=True)

    # 关闭子进程不需要的从端文件描述符
    os.close(slave_fd)

    # 存储输出的容器
    stdout_data = []
    stderr_data = []

    # 启动线程读取stdout（通过伪终端主端）
    stdout_thread = Thread(target=read_stdout, args=(master_fd, stdout_data))
    stdout_thread.daemon = True
    stdout_thread.start()

    # 启动线程读取stderr（通过管道）
    stderr_thread = Thread(target=read_stderr, args=(proc.stderr, stderr_data))
    stderr_thread.daemon = True
    stderr_thread.start()

    # 等待子进程结束
    proc.wait()

    # 关闭伪终端主端
    os.close(master_fd)

    # 等待线程结束
    stdout_thread.join()
    stderr_thread.join()

    # 合并结果
    final_stdout = get_final_text(''.join(stdout_data))
    final_stderr = get_final_text(''.join(stderr_data))

    return proc.returncode, final_stdout, final_stderr


class Runner:

    def __init__(self):
        self.cwd = os.getcwd()
        self.env = {}  # 初始化环境变量

    def __call__(self, cmd: str) -> (int, str, str):
        stdout = ""
        stderr = ""
        commands = cmd.split("&&")
        for c in commands:
            c = c.strip()
            # 处理cd命令
            if c.startswith("cd "):
                # 原有cd处理逻辑
                dir_name = c.split(' ', 1)[-1].strip()
                directory = os.path.join(self.cwd,
                                         os.path.expanduser(dir_name))
                if not os.path.exists(directory):
                    stderr += f"cd: no such directory: {directory}\n"
                    return (255, stdout, stderr)
                elif not os.path.isdir(directory):
                    stderr += f"cd: not a directory: {directory}\n"
                    return (255, stdout, stderr)
                self.cwd = directory
                continue
            # 处理export命令
            elif c.startswith("export "):
                rest = c[len('export '):].strip()
                if '=' in rest:
                    var_name, value = rest.split('=', 1)
                    var_name = var_name.strip()
                    value = value.strip()
                    # 去除值两侧的引号
                    if len(value) >= 2 and (value[0] == value[-1]
                                            and value[0] in ('"', "'")):
                        value = value[1:-1]
                    self.env[var_name] = value
                else:
                    # 处理export VAR（无赋值）
                    var_name = rest.strip()
                    if var_name in self.env:
                        # 保持原值，或设为空字符串
                        pass  # 这里不做修改，因为export VAR可能只是导出已存在的变量
                    else:
                        self.env[var_name] = ''
                continue
            # 执行其他命令
            (code, out, err) = run_command(c, cwd=self.cwd, extra_env=self.env)
            stdout += out
            stderr += err
            if code != 0:
                return (code, stdout, stderr)
        return (0, stdout, stderr)


def agent_print(*args, **kwargs):
    print("\033[94m", end="")
    sio = StringIO()
    print(*args, **kwargs, file=sio)
    sio.seek(0)
    for line in sio.readlines():
        print("[Agent]", line, end="")
    print("\033[0m", end="", flush=True)


def agent_print_cmd(cmd: str):
    print(f"\033[94m[Agent]\033[0m \033[93m{cmd}\033[0m", flush=True)


def agent_input(*args, **kwargs):
    print("\033[94m[Agent]\033[0m ", end="")
    print("\033[94m", end="")
    result = input(*args, **kwargs)
    print("\033[0m", end="", flush=True)
    return result


def get_final_text(input_str):
    # 设置足够大的行数和列数以避免截断
    screen = pyte.Screen(500, 3000)  # 列数，行数
    stream = pyte.Stream(screen)
    stream.feed(input_str)

    # 提取处理后的行并去除每行末尾的空格
    final_lines = [line.rstrip() for line in screen.display]
    # 合并行并用换行符连接，同时去除末尾的空白行
    return '\n'.join(final_lines).rstrip('\n')


def summary_print(messages: list):
    agent_print("SUMMARY")
    for message in messages:
        if message["role"] == "assistant":
            cmd = json.loads(message["content"])["cmd"].strip()
            if cmd != "":
                agent_print_cmd(cmd)


if __name__ == "__main__":
    main()
