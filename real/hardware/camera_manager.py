"""
camera_manager.py

Encapsulates the realsense_interface to provide a simple, high-level API
for managing a RealSense camera.

Key improvements for multi-modal time alignment:
- Dual timestamp design: cam_ts_hw (hardware) + cam_ts_mono (monotonic, for alignment)
- Adaptive field name resolution for robustness
- Default RGB output for ML frameworks
- Debug diagnostics on first run
"""

import time
import hashlib
import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Dict, Tuple, Any, List

# Assuming realsense_interface is in a location Python can find.
# If not, we may need to adjust sys.path.
from realsense_interface import SingleRealsense


# ==============================================================================
# Field Name Resolution (Adaptive to different SingleRealsense versions)
# ==============================================================================

# Possible field names for each metadata type
FIELD_ALIASES = {
    'hw_timestamp': [
        'camera_capture_timestamp',  # Current SingleRealsense
        'hw_timestamp',
        'capture_timestamp',
        'device_timestamp',
    ],
    'recv_timestamp': [
        'camera_receive_timestamp',  # Current SingleRealsense
        'recv_timestamp',
        'receive_timestamp',
        'host_timestamp',
    ],
    'frame_id': [
        'step_idx',  # Current SingleRealsense
        'frame_id',
        'frame_number',
        'sequence_id',
        'frame_idx',
    ],
    'timestamp': [
        'timestamp',  # Generic timestamp (usually = recv_timestamp)
    ],
}


def resolve_field(data: Dict[str, Any], aliases: List[str], default=None, warn_if_missing=True):
    """
    Try multiple possible field names to find a value.

    Args:
        data: Dictionary from SingleRealsense.get()
        aliases: List of possible field names to try (in priority order)
        default: Default value if none found
        warn_if_missing: Print warning if using default

    Returns:
        The found value or default
    """
    for field_name in aliases:
        if field_name in data:
            return data[field_name]
    if warn_if_missing and default is not None:
        # Only warn once per session (tracked via global set)
        if not hasattr(resolve_field, '_warned_fields'):
            resolve_field._warned_fields = set()
        field_key = aliases[0]  # Use primary name for tracking
        if field_key not in resolve_field._warned_fields:
            print(f"[CameraManager] Warning: Field {aliases} not found, using default: {default}")
            resolve_field._warned_fields.add(field_key)
    return default


# ==============================================================================
# Debug / Self-Check
# ==============================================================================

_debug_dump_done = False


def debug_dump_once(data: Dict[str, Any], selected_fields: Dict[str, Any]):
    """
    Print diagnostic information on first call (only once per session).

    Args:
        data: Raw dictionary from SingleRealsense.get()
        selected_fields: The fields we actually selected/used
    """
    global _debug_dump_done
    if _debug_dump_done:
        return
    _debug_dump_done = True

    print("\n" + "=" * 60)
    print("[CameraManager] SingleRealsense.get() Field Diagnostics")
    print("=" * 60)
    print(f"Available fields ({len(data)}):")
    for key in sorted(data.keys()):
        val = data[key]
        if isinstance(val, np.ndarray):
            print(f"  - {key}: ndarray {val.shape} {val.dtype}")
        elif isinstance(val, (int, float, str, bool)):
            print(f"  - {key}: {type(val).__name__} = {val}")
        else:
            print(f"  - {key}: {type(val).__name__}")
    print("\n" + "-" * 60)
    print("Selected field mappings:")
    for display_name, (actual_key, value) in selected_fields.items():
        if isinstance(value, float):
            print(f"  - {display_name}: '{actual_key}' = {value:.6f} s")
        elif isinstance(value, int):
            print(f"  - {display_name}: '{actual_key}' = {value}")
        else:
            print(f"  - {display_name}: '{actual_key}' = {value}")
    print("=" * 60 + "\n")


def compute_image_hash(image: np.ndarray) -> str:
    """
    Compute MD5 hash of image for deduplication.

    Note: This is a fallback when frame_id is unavailable.
    It has CPU overhead and should not be the primary method.

    Args:
        image: Image array (H, W, C)

    Returns:
        MD5 hex string
    """
    return hashlib.md5(image.tobytes()).hexdigest()


# ==============================================================================
# Main CameraManager Class
# ==============================================================================

class CameraManager:
    """
    A manager to handle a single RealSense camera, abstracting away the details of
    initialization, data fetching, and cleanup.

    Time alignment strategy:
    - cam_ts_hw: Hardware timestamp from RealSense (preserved for diagnostics, NOT for alignment)
    - cam_ts_mono: time.monotonic() timestamp when frame was retrieved (used for robot alignment)

    The cam_ts_mono is on the same clock domain as robot timestamps (both use time.monotonic()),
    enabling accurate cross-modal alignment.
    """

    def __init__(self, serial_number, width=1280, height=720, fps=30):
        """
        Initializes the CameraManager.

        Args:
            serial_number (str): The serial number of the RealSense camera.
                                Leave empty or None to auto-detect.
            width (int): The width of the color image.
            height (int): The height of the color image.
            fps (int): The capture frame rate.
        """
        # Allow empty serial_number for auto-detect
        if not serial_number:
            print("No serial number provided, will auto-detect camera...")
            self.serial_number = None
        else:
            self.serial_number = serial_number

        self.width = width
        self.height = height
        self.fps = fps

        self.shm_manager = None
        self.camera = None
        self.is_running = False
        self._intrinsics = None

    def start(self):
        """
        Initializes the SharedMemoryManager and the SingleRealsense camera instance.
        This starts the camera capture process in the background.
        """
        if self.is_running:
            print("Camera is already running.")
            return

        print(f"Starting camera {self.serial_number}...")
        try:
            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()

            # realsense_interface.SingleRealsense expects `resolution=(w, h)`
            # (not `color_width` / `color_height`).
            self.camera = SingleRealsense(
                shm_manager=self.shm_manager,
                serial_number=self.serial_number,
                resolution=(self.width, self.height),
                capture_fps=self.fps,
                enable_color=True,
                enable_depth=False,  # Keep disabled unless needed for the policy
                verbose=False,
            )
            self.camera.start()
            self.is_running = True
            print("Camera started successfully.")
        except Exception as e:
            print(f"Failed to start camera: {e}")
            if self.shm_manager:
                self.shm_manager.shutdown()
            raise

    def get_latest_frame(self, convert_to_rgb=True):
        """
        Fetches the latest color frame from the camera.

        Args:
            convert_to_rgb (bool): If True, converts the BGR image to RGB.

        Returns:
            np.ndarray: The color image frame, or None if not available.
        """
        if not self.is_running or self.camera is None:
            print("Error: Camera is not running.")
            return None

        data = self.camera.get()
        if 'color' in data:
            img = data['color']
            if convert_to_rgb:
                # RealSense outputs BGR, convert to RGB for most models
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        return None

    def get_latest_frame_with_meta(self, convert_to_rgb=True, enable_debug_dump=True):
        """
        Fetches the latest color frame WITH metadata for time alignment.

        This is the RECOMMENDED method for data collection.

        Time alignment strategy:
        - cam_ts_hw: Hardware timestamp from RealSense (diagnostic only, NOT for alignment)
        - cam_ts_mono: time.monotonic() when frame was retrieved (used for robot alignment)

        The cam_ts_mono is on the same clock domain as robot timestamps, enabling
        accurate cross-modal alignment. The hardware timestamp is preserved for
        diagnostics and to detect potential clock drift.

        Args:
            convert_to_rgb (bool): If True, converts BGR to RGB. Default: True (RGB for ML).
            enable_debug_dump (bool): Print field diagnostics on first call. Default: True.

        Returns:
            tuple: (image, metadata_dict) or (None, None) if unavailable.
                - image: np.ndarray, color image (BGR or RGB based on convert_to_rgb)
                - metadata: dict with keys:
                    - 'cam_ts_hw': hardware capture timestamp (seconds, RealSense global clock)
                    - 'cam_ts_mono': time.monotonic() at retrieval (seconds, for robot alignment)
                    - 'cam_ts_recv': host receive timestamp (seconds, time.time())
                    - 'frame_id': frame sequence number (int, or -1 if unavailable)
                    - 'color_space': 'BGR' or 'RGB'
                    - 'image_hash': MD5 hash of image (fallback for deduplication)

        Note on clock domains:
            RealSense hardware timestamps use a global clock that is NOT synchronized
            with the host's time.monotonic(). Always use cam_ts_mono for alignment
            with robot data. The hardware timestamp is useful for detecting clock drift
            or diagnosing timing issues.
        """
        if not self.is_running or self.camera is None:
            return None, None

        data = self.camera.get()
        if 'color' not in data:
            return None, None

        # Get image
        img = data['color']
        color_space = 'BGR'

        # Convert to RGB by default (most ML frameworks expect RGB)
        if convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color_space = 'RGB'

        # Record monotonic timestamp ASAP (same clock as robot)
        cam_ts_mono = time.monotonic()

        # Resolve fields with adaptive name matching
        hw_ts = resolve_field(data, FIELD_ALIASES['hw_timestamp'], default=0.0)
        recv_ts = resolve_field(data, FIELD_ALIASES['recv_timestamp'], default=time.time())
        frame_id = resolve_field(data, FIELD_ALIASES['frame_id'], default=-1)

        # Check if hw_ts is in milliseconds (old convention) and convert
        # RealSense reports in ms, SingleRealsense converts to s, but we handle both
        if hw_ts > 1e10:  # Likely milliseconds (timestamp > 300 years in seconds)
            hw_ts = hw_ts / 1000.0

        # Compute image hash for deduplication fallback
        img_hash = compute_image_hash(img)

        # Build metadata
        metadata = {
            'cam_ts_hw': float(hw_ts),           # Hardware timestamp (diagnostic only)
            'cam_ts_mono': float(cam_ts_mono),   # Monotonic timestamp (for robot alignment)
            'cam_ts_recv': float(recv_ts),       # Receive timestamp (time.time())
            'frame_id': int(frame_id),           # Frame sequence number
            'color_space': color_space,          # 'BGR' or 'RGB'
            'image_hash': img_hash,              # MD5 hash (deduplication fallback)
        }

        # Debug dump on first call
        if enable_debug_dump:
            selected_fields = {
                'hw_timestamp': (metadata['cam_ts_hw'], metadata['cam_ts_hw']),
                'recv_timestamp': (metadata['cam_ts_recv'], metadata['cam_ts_recv']),
                'frame_id': (metadata['frame_id'], metadata['frame_id']),
                'cam_ts_mono': (metadata['cam_ts_mono'], metadata['cam_ts_mono']),
            }
            # Find actual field names used
            actual_hw = None
            for alias in FIELD_ALIASES['hw_timestamp']:
                if alias in data:
                    actual_hw = alias
                    break
            actual_recv = None
            for alias in FIELD_ALIASES['recv_timestamp']:
                if alias in data:
                    actual_recv = alias
                    break
            actual_fid = None
            for alias in FIELD_ALIASES['frame_id']:
                if alias in data:
                    actual_fid = alias
                    break

            debug_info = {
                'hw_timestamp': (actual_hw or 'MISSING', metadata['cam_ts_hw']),
                'recv_timestamp': (actual_recv or 'MISSING', metadata['cam_ts_recv']),
                'frame_id': (actual_fid or 'MISSING', metadata['frame_id']),
                'cam_ts_mono': ('time.monotonic()', metadata['cam_ts_mono']),
            }
            debug_dump_once(data, debug_info)

        return img, metadata

    def get_intrinsics(self):
        """
        Gets the camera intrinsics matrix.

        Returns:
            np.ndarray: The 3x3 camera intrinsics matrix.
        """
        if not self.is_running or self.camera is None:
            print("Error: Camera is not running. Cannot get intrinsics.")
            return None

        if self._intrinsics is None:
            self._intrinsics = self.camera.get_intrinsics()

        return self._intrinsics

    def stop(self):
        """
        Stops the camera and cleans up resources.
        """
        if not self.is_running:
            return

        print("Stopping camera...")
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

        if self.shm_manager is not None:
            self.shm_manager.shutdown()
            self.shm_manager = None

        self.is_running = False
        print("Camera stopped.")

    @staticmethod
    def find_available_cameras():
        """
        Static method to find all connected RealSense camera serial numbers.

        Returns:
            list: A list of serial numbers for connected devices.
        """
        return SingleRealsense.get_connected_devices_serial()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
