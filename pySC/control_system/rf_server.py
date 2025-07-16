from .send_receive import pySCServerError, send_int, send_float

def rf_server(conn, signal, SC):
    variable = signal.split(' ')[1]
    command = signal[:3]
    server, device, prop = variable.strip().split('/')
    device = device.lower()
    if device in SC.rf_settings.systems:
        if command == 'GET':
            if prop == 'VOLTAGE':
                setpoint = SC.rf_settings.systems[device].voltage
            if prop == 'PHASE':
                setpoint = SC.rf_settings.systems[device].phase
            if prop == 'FREQUENCY':
                setpoint = SC.rf_settings.systems[device].frequency
            send_int(conn, 3) # 2 for float
            send_float(conn, setpoint)
        elif command == 'SET':
            value = float(signal.split(' ')[2])
            if prop == 'VOLTAGE':
                SC.rf_settings.systems[device].set_voltage(value)
            if prop == 'PHASE':
                SC.rf_settings.systems[device].set_phase(value)
            if prop == 'FREQUENCY':
                SC.rf_settings.systems[device].set_frequency(value)
            send_int(conn, 1) # 1 for set ok
    else:
        raise pySCServerError 