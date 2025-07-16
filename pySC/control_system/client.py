import socket
from .send_receive import receive_int, receive_float, receive_nparray

def read(address: str):
    # Address format is '127.0.0.1:1313/ORBIT/RAW/X'
    host, port = address.split('/')[0].split(':')
    variable = '/'.join(address.split('/')[1:])
    signal = f"GET {variable}"

    s = socket.socket()
    s.connect((host,int(port)))
    s.sendall(signal.encode())

    code = receive_int(s)
    if code == 0:
        print('ERROR during read.')
        value = None
    elif code == 1:
        value = None
    elif code == 2:
        value = receive_int(s)
    elif code == 3:
        value = receive_float(s)
    elif code == 4:
        value = receive_nparray(s)

    s.close()
    return value

def write(address: str, value: float):
    # Address format is '127.0.0.1:1313/ORBIT/RAW/X'
    host, port = address.split('/')[0].split(':')
    variable = '/'.join(address.split('/')[1:])
    signal = f"SET {variable} {str(value)}"

    s = socket.socket()
    s.connect((host,int(port)))
    s.sendall(signal.encode())

    code = receive_int(s)
    if code == 0:
        print('ERROR during read.')
    elif code == 1:
        pass
    else:
        print(f'ERROR: Unknown code {code} during write.')

    s.close()
    return None