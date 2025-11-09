from enum import IntEnum

class MeasurementCode(IntEnum):
    INITIALIZED = 0
    HYSTERESIS = 1
    HYSTERESIS_DONE = 2

class BBACode(IntEnum):
    HYSTERESIS = MeasurementCode.HYSTERESIS.value
    HYSTERESIS_DONE = MeasurementCode.HYSTERESIS_DONE.value
    HORIZONTAL = 3
    HORIZONTAL_DONE = 4
    VERTICAL = 5
    VERTICAL_DONE = 6
    DONE = 7

class ResponseCode(IntEnum):
    INITIALIZED = MeasurementCode.INITIALIZED.value
    MEASURING = 3
    DONE = 4

class DispersionCode(IntEnum):
    INITIALIZED = MeasurementCode.INITIALIZED.value
    MEASURING = 3
    DONE = 4