"""
RealSense Interface Package

High-performance Intel RealSense camera interface with multi-camera support.
Extracted from diffusion_policy project.
"""

from .single_realsense import SingleRealsense
from .multi_realsense import MultiRealsense
from .video_recorder import VideoRecorder
from ._version import __version__, __author__, __description__

__all__ = ["SingleRealsense", "MultiRealsense", "VideoRecorder"]