import base64
from typing import Optional

import numpy as np

from .models import ArrayPayload


def array_to_payload(array: Optional[np.ndarray], include_data: bool = True) -> Optional[ArrayPayload]:
    if array is None:
        return None

    arr = np.asarray(array)
    contiguous = np.ascontiguousarray(arr)
    data = None
    if include_data:
        data = base64.b64encode(contiguous.tobytes(order="C")).decode("ascii")

    return ArrayPayload(
        shape=tuple(int(x) for x in contiguous.shape),
        dtype=str(contiguous.dtype),
        data=data,
        order="C",
    )


def array_from_payload(payload: ArrayPayload) -> np.ndarray:
    if payload.data is None:
        raise ValueError("payload has no data")
    raw = base64.b64decode(payload.data.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.dtype(payload.dtype))
    return arr.reshape(payload.shape, order=payload.order)


def flatten_time_frequency(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a time-major 1D array matching MarineTargetTracker SpecData layout."""
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError("expected a 2D time-frequency matrix")
    return np.ascontiguousarray(arr).reshape(-1)
