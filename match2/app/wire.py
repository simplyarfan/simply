"""One JSON message each way over a unix socket. Shared by app.py and server.py."""

import json
import os
import socket

SOCKET_PATH = os.environ.get("SIMPLY_SOCKET", "/tmp/simply.sock")


def send(sock: socket.socket, obj) -> None:
    sock.sendall(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")


def recv(sock: socket.socket):
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    if not buf:
        raise ConnectionError("server closed the connection without answering")
    return json.loads(buf.decode("utf-8"))


def ask(obj, timeout: float):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(SOCKET_PATH)
        send(s, obj)
        return recv(s)
