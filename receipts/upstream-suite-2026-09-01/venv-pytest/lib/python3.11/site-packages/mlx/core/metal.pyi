"""mlx.metal"""



def is_available() -> bool:
    """Check if the Metal back-end is available."""

def get_active_memory() -> int: ...

def get_peak_memory() -> int: ...

def reset_peak_memory() -> None: ...

def get_cache_memory() -> int: ...

def set_memory_limit(limit: int) -> int: ...

def set_cache_limit(limit: int) -> int: ...

def set_wired_limit(limit: int) -> int: ...

def clear_cache() -> None: ...

def start_capture(path: str) -> None:
    """
    Start a Metal capture.

    Args:
      path (str): The path to save the capture which should have
        the extension ``.gputrace``.
    """

def stop_capture() -> None:
    """Stop a Metal capture."""

def device_info() -> dict[str, str | int]: ...
