import os
import sys
import pty
import time
import fcntl
import termios
import struct
import pexpect
import argparse
import subprocess
import json
from io import StringIO
from abc import ABC, abstractmethod
import getpass
from threading import Thread, Event

from openai import OpenAI
import pyte


class Runner:

    def __init__(self):
        self.proc = pexpect.spawn("bash",
                                  echo=True,
                                  env=os.environ.update({"TERM": "linux"}))
        self.__update_win_size()
        # end_str 不要直接写出来，不然 cat main.py 会出问题
        self.end_str = "\x5f\x5f\x41\x55\x54\x4f\x52\x4d\x49\x4e\x41\x4c\x5f\x45\x4e\x44\x5f\x5f"
        self.stop = Event()

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
        count = 2
        while True:
            try:
                data = self.proc.read_nonblocking(1024, timeout=0)
            except Exception:
                if count == 0:
                    break
                count -= 1
                time.sleep(0.1)
                continue
            if data:
                result += data
        return result

    def __copy_output(self) -> (int, bytes):
        end_len = len(self.end_str)
        output = b''
        while True:
            data = self.__read_output_all()
            if len(data) == 0:
                continue
            if bytes(self.end_str, encoding="utf-8") not in data:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
                output += data
            else:
                break

        data, meta_info = data.split(bytes(self.end_str, encoding="utf-8"), 1)
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
        output += data
        ret_code = int(
            meta_info.split(bytes(self.end_str, encoding="utf-8"),
                            1)[0].decode())
        return ret_code, output

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

    def __call__(self, cmd: str) -> (int, str):
        self.__update_win_size()
        self.__read_output_all()

        cmd = bytes(f"{cmd} ; echo {self.end_str}\"$?\"{self.end_str}\n",
                    encoding="utf-8")
        self.proc.send(cmd)
        self.proc.read(len(cmd) + 1)
        settings = self.__setup_terminal()

        self.stop.clear()
        thread = Thread(target=self.__copy_stdin)
        thread.daemon = True
        thread.start()
        ret_code, output = self.__copy_output()
        self.stop.set()
        self.__restore_terminal(settings)
        output = self.__get_final_text(output)
        thread.join()
        return ret_code, output


runner = Runner()
while True:
    runner(input('> '))
