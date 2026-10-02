"""Regression coverage for device names in the socket server dispatch path."""

import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from pySC.control_system import server
from pySC.control_system.send_receive import INT_SIZE


class RequestsComplete(Exception):
    """Stop the server once all test requests have been handled."""


@pytest.fixture
def serve_request(monkeypatch):
    def run(signal, sc):
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.recv.return_value = signal.encode()
        listener = MagicMock()
        listener.__enter__.return_value = listener
        listener.accept.side_effect = [
            (conn, ("127.0.0.1", 12345)),
            RequestsComplete(),
        ]
        monkeypatch.setattr(server.socket, "socket", Mock(return_value=listener))
        monkeypatch.setattr(server, "atexit", SimpleNamespace(register=Mock()))
        # Keep the refresh deadline fixed so these tests need no real-time waits.
        monkeypatch.setattr(server, "time", SimpleNamespace(time=lambda: 0.0))
        sc.bpm_system = SimpleNamespace(capture_orbit=Mock(return_value=([], [])))

        with pytest.raises(RequestsComplete):
            server.start_server(sc)
        return conn

    return run


def assert_response(conn, command, value):
    if command == "GET":
        payload = struct.pack("=d", value)
        expected = [
            call((3).to_bytes(INT_SIZE, "big")),
            call(len(payload).to_bytes(INT_SIZE, "big")),
            call(payload),
        ]
    else:
        expected = [call((1).to_bytes(INT_SIZE, "big"))]
    assert conn.sendall.call_args_list == expected


@pytest.mark.regression
@pytest.mark.parametrize("device", ["q1", "cell/q1", "ring/cell/q1"])
@pytest.mark.parametrize("command", ["GET", "SET"])
def test_magnet_device_names(serve_request, device, command):
    control = f"{device}/B2"
    settings = SimpleNamespace(
        controls={control: object()}, get=Mock(return_value=2.5), set=Mock()
    )
    sc = SimpleNamespace(magnet_settings=settings)
    signal = f"{command} MAGNET/{control}"
    if command == "SET":
        signal += " 3.5"

    conn = serve_request(signal, sc)

    if command == "GET":
        settings.get.assert_called_once_with(control)
        settings.set.assert_not_called()
    else:
        settings.set.assert_called_once_with(control, 3.5)
        settings.get.assert_not_called()
    assert_response(conn, command, 2.5)


@pytest.mark.regression
@pytest.mark.parametrize("device", ["cavity", "cell/cavity", "ring/cell/cavity"])
@pytest.mark.parametrize("command", ["GET", "SET"])
@pytest.mark.parametrize("prop", ["VOLTAGE", "PHASE", "FREQUENCY"])
def test_rf_device_names(serve_request, device, command, prop):
    values = {"voltage": 2.5, "phase": 0.25, "frequency": 500e6}
    setters = {f"set_{name}": Mock() for name in values}
    system = SimpleNamespace(**values, **setters)
    sc = SimpleNamespace(rf_settings=SimpleNamespace(systems={device: system}))
    signal = f"{command} RF/{device}/{prop}"
    if command == "SET":
        signal += " 3.5"

    conn = serve_request(signal, sc)

    for name, setter in setters.items():
        if command == "SET" and name == f"set_{prop.lower()}":
            setter.assert_called_once_with(3.5)
        else:
            setter.assert_not_called()
    assert_response(conn, command, values[prop.lower()])
