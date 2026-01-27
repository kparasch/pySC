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
    AFTER_SET = 3
    AFTER_GET = 4
    AFTER_RESTORE = 5
    MEASURING = 5 # same as AFTER_RESTORE on purpose
    DONE = 6

class DispersionCode(IntEnum):
    INITIALIZED = MeasurementCode.INITIALIZED.value
    MEASURING = 3
    DONE = 4