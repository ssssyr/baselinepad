from typing import List, Tuple, Optional, Dict
import math
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: List[float],  
    start_time: float, 
    dt: float, 
    eps:float=1e-5,
    next_global_idx: Optional[int]=0,
    allow_negative=False
    ) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx. 
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt 
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


def align_timestamps(    
        timestamps: List[float], 
        target_global_idxs: List[int], 
        start_time: float, 
        dt: float, 
        eps:float=1e-5):
    if isinstance(target_global_idxs, np.ndarray):
        target_global_idxs = target_global_idxs.tolist()
    assert len(target_global_idxs) > 0

    local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
        timestamps=timestamps,
        start_time=start_time,
        dt=dt,
        eps=eps,
        next_global_idx=target_global_idxs[0],
        allow_negative=True
    )
    if len(global_idxs) > len(target_global_idxs):
        # if more steps available, truncate
        global_idxs = global_idxs[:len(target_global_idxs)]
        local_idxs = local_idxs[:len(target_global_idxs)]
    
    if len(global_idxs) == 0:
        import pdb; pdb.set_trace()

    for i in range(len(target_global_idxs) - len(global_idxs)):
        # if missing, repeat
        local_idxs.append(len(timestamps)-1)
        global_idxs.append(global_idxs[-1] + 1)
    assert global_idxs == target_global_idxs
    assert len(local_idxs) == len(global_idxs)
    return local_idxs