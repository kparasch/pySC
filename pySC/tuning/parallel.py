import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
from multiprocessing import Queue
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

def get_listener_and_queue(logger):
    log_queue = Queue()
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    listener = QueueListener(log_queue, handler, respect_handler_level=True)
    return listener, log_queue

def logging_init(queue):
    """Initialize logging for the worker process."""
    # Get the root logger
    root_logger = logging.getLogger()

    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    queue_handler = QueueHandler(queue)
    pid = multiprocessing.current_process().pid
    formatter = logging.Formatter(fmt="{asctime} | {levelname}" + f" | PID: {pid} | " + "{message}",
                                  datefmt="%d %b %Y, %H:%M:%S",
                                  style='{')
    queue_handler.setFormatter(formatter)
    root_logger.addHandler(queue_handler)

def parallel_tbba_target(SC_model, SC_class, bpm_names, shots_per_trajectory, n_corr_steps, queue, log_queue):
    logging_init(log_queue)
    SC: "SimulatedCommissioning" = SC_class.model_validate(SC_model)
    offsets_x, offsets_y = SC.tuning.do_trajectory_bba(bpm_names=bpm_names, shots_per_trajectory=shots_per_trajectory, n_corr_steps=n_corr_steps)
    queue.put((bpm_names, offsets_x, offsets_y))
    del SC

def parallel_obba_target(SC_model, SC_class, bpm_names, shots_per_orbit, n_corr_steps, queue, log_queue):
    logging_init(log_queue)
    SC: "SimulatedCommissioning" = SC_class.model_validate(SC_model)
    offsets_x, offsets_y = SC.tuning.do_orbit_bba(bpm_names=bpm_names, shots_per_orbit=shots_per_orbit, n_corr_steps=n_corr_steps)
    queue.put((bpm_names, offsets_x, offsets_y))
    del SC