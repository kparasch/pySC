from .send_receive import pySCServerError, send_int, send_float

def magnet_server(conn, signal, SC):
    variable = signal.split(' ')[1]
    command = signal[:3]
    server, _, device_prop = variable.strip().partition('/')
    device, _, prop = device_prop.rpartition('/')

    control = '/'.join([device, prop])
    if control not in SC.magnet_settings.controls.keys():
        raise pySCServerError

    if command == 'GET':
        setpoint = SC.magnet_settings.get(control)
        send_int(conn, 3) # 2 for float
        send_float(conn, setpoint)
    elif command == 'SET':
        value = float(signal.split(' ')[2])
        SC.magnet_settings.set(control, value)
        send_int(conn, 1) # 1 for set ok
    else:
        raise pySCServerError
