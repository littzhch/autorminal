import pexpect
import sys
import os
import threading
import termios
import tty

shell = pexpect.spawn("bash", timeout=None)


def setup_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] &= ~termios.ICANON
    new_settings[3] &= ~termios.ECHO
    os.set_blocking(fd, False)
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
    return old_settings


def copy_stdin():
    while True:
        ch = sys.stdin.read(1)
        if ch != '':
            shell.send(ch)


setup_terminal()

t = threading.Thread(target=copy_stdin)
t.daemon = True
t.start()

while True:
    sys.stdout.buffer.write(shell.read_nonblocking(1024))
    sys.stdout.flush()
