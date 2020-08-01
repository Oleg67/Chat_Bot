import json
import time
import socket
import pathlib
import pickle
from functools import wraps
from logging import getLogger
from typing import Any

import numpy

from config import TRAINED_DATA_PATH, MODEL_TYPE


log = getLogger(__name__)


def path_to_all_pkl_files(
    root_path: str = TRAINED_DATA_PATH, model_type: str = MODEL_TYPE, default_pattern: str = "*.pkl"
) -> pathlib.Path.rglob:
    return pathlib.Path(root_path, model_type).rglob(default_pattern)


def read_one_pkl_file(path: str, log_message: bool = True) -> Any:
    with open(file=path, mode="rb") as file:
        data = pickle.loads(file.read())
        if log_message:
            log.debug(f"File: {file.name} unpickled successfully")
        return data


def log_execution_time(func):
    """Only for sync functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(f"<{func.__module__}.{func.__name__}> execution time {round(end - start, 3)} seconds")
        return result

    return wrapper


def uniq_id_from_time() -> int:
    return int(time.time() * 1000)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_port(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except Exception:
        return False
