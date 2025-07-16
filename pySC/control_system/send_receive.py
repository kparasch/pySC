import numpy as np
from typing import Optional
import socket

INT_SIZE = 8

class pySCServerError(Exception):
    pass

def send_int(sock: socket.socket, i: int) -> None:
    int_bytes = i.to_bytes(INT_SIZE, 'big')
    sock.sendall(int_bytes)

def receive_int(sock: socket.socket) -> Optional[int]:
    int_bytes = sock.recv(INT_SIZE)
    if not int_bytes:
        return None
    i = int.from_bytes(int_bytes, 'big')
    return i

def send_float(sock: socket.socket, f: float) -> None:
    float_bytes = np.float64(f).tobytes()
    send_int(sock, len(float_bytes))
    sock.sendall(float_bytes)

def receive_float(sock: socket.socket) -> Optional[float]:
    buffer_size = receive_int(sock)
    if buffer_size is None:
        return None
    float_bytes = sock.recv(buffer_size)
    if not float_bytes:
        return None
    f = float(np.frombuffer(float_bytes)[0])
    return f

def send_nparray(sock: socket.socket, arr: np.ndarray, ndim: int = 1) -> None:
    arr_bytes = arr.tobytes()
    shape = arr.shape
    ndim = len(shape)
    # send number of dimensions
    send_int(sock, ndim)

    # send shape
    for length in shape:
        send_int(sock, length)
    
    # send buffer size
    send_int(sock, len(arr_bytes))

    # send buffer
    sock.sendall(arr_bytes)

def receive_nparray(sock: socket.socket) -> Optional[np.ndarray]:
    # receive number of dimensions
    ndim = receive_int(sock)
    if ndim is None:
        return None

    # receive shape
    shape = []
    for ii in range(ndim):
        length = receive_int(sock)
        if length is None:
            return None
        shape.append(length)
    
    # receive buffer size
    buffer_size = receive_int(sock)
    if buffer_size is None:
        return None

    # receive data
    arr_bytes = sock.recv(buffer_size)
    if len(arr_bytes) != buffer_size:
        raise Exception(f"Received {len(arr_bytes)} bytes, expected {buffer_size} bytes")
    if not arr_bytes:
        return None

    arr = np.frombuffer(arr_bytes, dtype=np.float64)
    arr = arr.reshape(shape)
    return arr