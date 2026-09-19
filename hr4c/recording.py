"""ACT-compatible episode writer with atomic publication."""

from __future__ import annotations

import os
import numpy as np


def write_episode(path, observations, actions, metadata=None):
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise RuntimeError("h5py is required to write ACT episodes") from error
    if not observations or len(observations) != len(actions):
        raise ValueError("observations and actions must have equal, non-zero length")
    cameras = tuple(observations[0].images)
    if any(tuple(item.images) != cameras for item in observations):
        raise ValueError("camera set/order changed during the episode")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with h5py.File(temporary, "w") as root:
            root.attrs["sim"] = False
            root.attrs["schema_version"] = 1
            for key, value in (metadata or {}).items():
                root.attrs[key] = value
            root.create_dataset("action", data=np.asarray(actions))
            group = root.create_group("observations")
            group.create_dataset("qpos", data=np.asarray([item.qpos for item in observations]))
            group.create_dataset("timestamp", data=np.asarray([item.timestamp for item in observations]))
            image_group = group.create_group("images")
            image_time_group = group.create_group("image_timestamps")
            for camera in cameras:
                image_group.create_dataset(camera, data=np.asarray([item.images[camera] for item in observations]),
                                           compression="gzip", compression_opts=1)
                image_time_group.create_dataset(camera, data=np.asarray(
                    [item.image_timestamps[camera] for item in observations]))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_episode_npz(path, observations, actions):
    """Dependency-free diagnostic recording; not an ACT training artifact."""
    if not observations or len(observations) != len(actions):
        raise ValueError("observations and actions must have equal, non-zero length")
    cameras = tuple(observations[0].images)
    payload = {
        "action": np.asarray(actions),
        "qpos": np.asarray([item.qpos for item in observations]),
        "timestamp": np.asarray([item.timestamp for item in observations]),
    }
    for camera in cameras:
        payload[f"image_{camera}"] = np.asarray([item.images[camera] for item in observations])
        payload[f"image_timestamp_{camera}"] = np.asarray(
            [item.image_timestamps[camera] for item in observations])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
