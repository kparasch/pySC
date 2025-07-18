import socket
import numpy as np
from .send_receive import send_int, send_nparray, pySCServerError

def orbit_server(conn: socket.socket, signal: str, orbit_x: np.ndarray, orbit_y: np.ndarray, SC):
    variable = signal.split(' ')[1]
    command = signal[:3]
    if variable == 'ORBIT/RAW/X':
        if command == 'GET':
            send_int(conn, 4) # 4 for nparray
            send_nparray(conn, orbit_x)
        else:
            send_int(conn, 0)
    elif variable == 'ORBIT/RAW/Y':
        if command == 'GET':
            send_int(conn, 4) # 4 for nparray
            send_nparray(conn, orbit_y)
        else:
            raise pySCServerError 
    if variable == 'ORBIT/INJECTION/CORRECT':
        if command == 'SET':
            value = float(signal.split(' ')[2])
            SC.tuning.correct_injection(parameter=value)
    else:
        raise pySCServerError 