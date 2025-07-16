import socket
import atexit
import time
from typing import TYPE_CHECKING

from .send_receive import send_int, pySCServerError
from .orbit_server import orbit_server
from .magnet_server import magnet_server
from .rf_server import rf_server

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
# PORT = 13131  # Port to listen on (non-privileged ports are > 1023)

def start_server(SC: "SimulatedCommissioning" , port : int = 13131, refresh_rate : float = 1):

    orbit_x, orbit_y = SC.bpm_system.capture_orbit()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, port))
        s.listen()
        s.settimeout(refresh_rate)
        print("Socket successfully created")
        atexit.register(s.close)
        while True:
            print('Calculating orbit...')
            orbit_x, orbit_y = SC.bpm_system.capture_orbit()
            start_time = time.time()
            print('Accepting commands...')
            while True:
                try:
                    ## timeout normally refreshes everytime a new connection is accepted
                    ## this is an extra check to raise a timeout error if the server has
                    ## been accepting commands for more than 'timeout'.
                    if time.time() > start_time + refresh_rate:
                        raise socket.timeout

                    conn, addr = s.accept()
                    with conn:
                        print(f"Connected by {addr}")
                        data = conn.recv(1024)
                        signal = data.decode()
                        try: 
                            print(f"Received: \'{signal}\'")
                            if len(signal) > 2 and signal[:3] in ['GET', 'SET']:

                                variable = signal.split(' ')[1]
                                server, device, prop = variable.strip().split('/')

                                if server == 'ORBIT':
                                    orbit_server(conn, signal, orbit_x, orbit_y)

                                if server == 'MAGNET':
                                    magnet_server(conn, signal, SC)

                                if server == 'RF':
                                    rf_server(conn, signal, SC)

                        except pySCServerError:
                            send_int(conn, 0)
                            raise socket.timeout
                except socket.timeout:
                    break

