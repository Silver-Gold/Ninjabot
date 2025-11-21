"""
Fruit Ninja CV: centroid tracking + direction classification (scale-invariant)
with automatic multi-fruit mouse swipes.

Behavior:
- Green boxes: raw motion blobs (movement detected).
- Red boxes: tracks classified as upward-moving fruits.
- For each frame, any newly-classified fruits are sliced in ONE continuous
  mouse drag passing near the tops of their bounding boxes.
- Swipe duration is configurable and can be tuned very small.
- Press ESC at any time (any window) to quit.

Dependencies:
    pip install pynput mss opencv-python numpy
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import mss
import numpy as np
from pynput import keyboard, mouse


# -------------
# CONFIGURATION
# -------------

CONFIRM_KEY = keyboard.Key.space
CANCEL_KEY = keyboard.Key.esc

# ---- Motion / blob detection ----
FRAME_DIFF_THRESH = 18        # grayscale diff threshold for motion mask
KERNEL_SIZE = 5               # morphology kernel size

# ---- Scale-invariant blob filters ----
MIN_AREA_RATIO = 0.005        # min blob area as fraction of frame area
MIN_BOX_RATIO = 0.07          # min bbox width/height as fraction of min(frame_w, frame_h)

# ---- Tracking ----
MAX_ASSOC_DIST_RATIO = 0.15   # max distance to link detection to a track (fraction of frame diagonal)
MIN_HISTORY_FOR_CLASS = 3     # need at least this many positions to classify direction
MAX_HISTORY_LEN = 10          # store up to this many positions per track
MAX_MISSED_FRAMES = 5         # delete track if not seen for this many frames

# ---- Direction classification (scale-invariant) ----
# dy_norm = dy / frame_height (negative = up)
DY_UP_THRESH_NORM = 0.01      # mean(dy_norm) must be < -DY_UP_THRESH_NORM to be "upward"
# dx_norm = dx / frame_width
DX_MAX_FRUIT_NORM = 0.02      # fruits: average |dx|/frame_width must be below this (slices can be more horizontal)

# ---- Spatial heuristics ----
BOTTOM_REGION_MIN_RATIO = 0.35   # only consider tracks whose centroid_y > this * frame_height
BOTTOM_REGION_MAX_RATIO = 0.95   # and centroid_y < this * frame_height

# ---- Swipe behavior ----
SWIPE_LENGTH_RATIO = 0.25       # base slash length as fraction of min(frame_w, frame_h)
SWIPE_DURATION_MS = 40          # total duration of a full multi-fruit swipe (tweak this)
SWIPE_STEPS_PER_SEG = 4         # interpolation steps per segment between fruits

SWIPE_COOLDOWN_FRAMES = 2   # minimum frames between swipes (tune this)

# Global exit flag controlled by ESC
exit_requested = False


# ---------
# TRACKING
# ---------

@dataclass
class Track:
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    last_seen_frame: int = 0
    has_swiped: bool = False  # ensure we only trigger one swipe per track

    def add_detection(self, cx: float, cy: float, bbox: Tuple[int, int, int, int], frame_index: int):
        self.positions.append((cx, cy))
        if len(self.positions) > MAX_HISTORY_LEN:
            self.positions.pop(0)
        self.bbox = bbox
        self.last_seen_frame = frame_index

    def is_upward_fruit(
        self,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Decide if this track looks like an upward-moving fruit."""
        if len(self.positions) < MIN_HISTORY_FOR_CLASS:
            return False

        # Use the last K steps (up to MIN_HISTORY_FOR_CLASS)
        k = min(MIN_HISTORY_FOR_CLASS, len(self.positions) - 1)
        recent = self.positions[-(k + 1):]  # k+1 points -> k steps

        dys = []
        dxs = []
        for (x_prev, y_prev), (x_curr, y_curr) in zip(recent[:-1], recent[1:]):
            dxs.append(x_curr - x_prev)
            dys.append(y_curr - y_prev)

        if not dys:
            return False

        # Normalize by frame dimensions
        dy_norm = np.array(dys) / float(frame_h)
        dx_norm = np.array(dxs) / float(frame_w)

        mean_dy_norm = float(np.mean(dy_norm))
        mean_abs_dx_norm = float(np.mean(np.abs(dx_norm)))

        # Centroid position (current)
        cx, cy = self.positions[-1]

        # Only consider objects in bottom region (where fruits are launched)
        if not (BOTTOM_REGION_MIN_RATIO * frame_h <= cy <= BOTTOM_REGION_MAX_RATIO * frame_h):
            return False

        # Require net upward motion (negative dy, big enough in magnitude)
        if mean_dy_norm >= -DY_UP_THRESH_NORM:
            return False

        # Require mostly vertical motion (low horizontal component)
        if mean_abs_dx_norm > DX_MAX_FRUIT_NORM:
            return False

        # Also reject if the latest step is clearly downward (changing direction)
        latest_dy_norm = dy_norm[-1]
        if latest_dy_norm > 0.0:
            return False

        return True


class Tracker:
    def __init__(self, frame_w: int, frame_h: int):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.frame_area = float(frame_w * frame_h)
        self.frame_diag = float(math.hypot(frame_w, frame_h))
        self.min_side = float(min(frame_w, frame_h))

    def _distance(self, p1, p2) -> float:
        (x1, y1), (x2, y2) = p1, p2
        return math.hypot(x1 - x2, y1 - y2)

    def update(
        self,
        detections: List[Tuple[float, float, Tuple[int, int, int, int]]],
        frame_index: int,
    ):
        """
        detections: list of (cx, cy, bbox)
        """
        # Associate detections to existing tracks using greedy nearest neighbor
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())

        max_dist = MAX_ASSOC_DIST_RATIO * self.frame_diag

        # Precompute distances
        distances: List[Tuple[float, int, int]] = []  # (dist, track_id, det_idx)
        for track_id in self.tracks:
            track = self.tracks[track_id]
            tx, ty = track.positions[-1] if track.positions else (None, None)
            if tx is None:
                continue
            for det_idx, (cx, cy, _) in enumerate(detections):
                d = self._distance((tx, ty), (cx, cy))
                if d <= max_dist:
                    distances.append((d, track_id, det_idx))

        # Sort by distance so we assign closest matches first
        distances.sort(key=lambda x: x[0])

        for d, track_id, det_idx in distances:
            if (track_id not in unmatched_tracks) or (det_idx not in unmatched_detections):
                continue
            # Assign
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_idx)

            cx, cy, bbox = detections[det_idx]
            self.tracks[track_id].add_detection(cx, cy, bbox, frame_index)

        # Create new tracks for remaining detections
        for det_idx in unmatched_detections:
            cx, cy, bbox = detections[det_idx]
            track = Track(track_id=self.next_id)
            track.add_detection(cx, cy, bbox, frame_index)
            self.tracks[self.next_id] = track
            self.next_id += 1

        # Remove stale tracks
        to_delete = [
            track_id
            for track_id, tr in self.tracks.items()
            if frame_index - tr.last_seen_frame > MAX_MISSED_FRAMES
        ]
        for track_id in to_delete:
            del self.tracks[track_id]

    def get_upward_fruit_tracks(self) -> List[Track]:
        return [
            tr
            for tr in self.tracks.values()
            if tr.is_upward_fruit(self.frame_w, self.frame_h)
        ]


# ---------------------
# REGION CONFIGURATION
# ---------------------

def select_play_area() -> Dict[str, int]:
    """Interactively select the Fruit Ninja play area with the mouse + SPACE."""
    points: List[Tuple[int, int]] = []
    mouse_controller = mouse.Controller()

    print("\n=== Configure Play Area ===")
    print("Hover over the FIRST corner and press SPACE.")
    print("Then hover over the OPPOSITE corner and press SPACE again.")
    print("Press ESC at any time to cancel.\n")

    def on_press(key):
        nonlocal points

        if key == CANCEL_KEY:
            print("Selection cancelled (ESC pressed).")
            return False

        if key == CONFIRM_KEY:
            x, y = mouse_controller.position
            points.append((x, y))
            print(f"Captured point {len(points)} at ({x}, {y})")

            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                left = int(min(x1, x2))
                top = int(min(y1, y2))
                right = int(max(x1, x2))
                bottom = int(max(y1, y2))

                width = right - left
                height = bottom - top

                print("\n=== Play Area Bounding Box ===")
                print(f"Top-left:     ({left}, {top})")
                print(f"Bottom-right: ({right}, {bottom})")
                print(f"Width:  {width}")
                print(f"Height: {height}")
                print("================================\n")

                return False  # stop listener

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    if len(points) != 2:
        return {}

    (x1, y1), (x2, y2) = points
    left = int(min(x1, x2))
    top = int(min(y1, y2))
    right = int(max(x1, x2))
    bottom = int(max(y1, y2))

    region = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
    }

    return region


# --------------
# GLOBAL ESC HANDLER
# --------------

def on_global_key_press(key):
    """Global ESC handler: sets exit_requested so loops can stop."""
    global exit_requested
    if key == keyboard.Key.esc:
        exit_requested = True
        print("ESC pressed: exit requested.")
        # Returning False stops this listener; that's fine because we only need one press
        return False


# --------------
# MULTI-FRUIT SWIPE
# --------------

def perform_multi_swipe(
    mouse_controller: mouse.Controller,
    points: List[Tuple[float, float]],
    swipe_length: int,
    duration_ms: int = SWIPE_DURATION_MS,
    steps_per_segment: int = SWIPE_STEPS_PER_SEG,
):
    """
    Perform one continuous mouse drag that passes near all given points,
    biased to move from top to bottom overall.

    - points: list of (screen_x, screen_y) targets (already in screen coords)
    - swipe_length: used to extend the swipe above/below fruits
    - duration_ms: total swipe duration for the entire path
    - steps_per_segment: how many small moves per segment between points
    """
    if not points:
        return

    # Sort by Y so we go roughly top -> bottom
    points = sorted(points, key=lambda p: p[1])

    first_x, first_y = points[0]
    last_x, last_y = points[-1]

    L = float(swipe_length)

    # Start a bit ABOVE the highest fruit and end a bit BELOW the lowest
    start = (first_x, first_y - L / 2.0)
    end = (last_x,  last_y + L / 2.0)

    path = [start] + points + [end]

    if len(path) < 2:
        return

    total_segments = len(path) - 1
    if total_segments <= 0:
        return

    if steps_per_segment <= 0:
        steps_per_segment = 1

    total_steps = max(1, total_segments * steps_per_segment)
    dt = (duration_ms / 1000.0) / float(total_steps)

    # Start at first point
    x0, y0 = path[0]
    mouse_controller.position = (int(x0), int(y0))
    time.sleep(0.003)
    mouse_controller.press(mouse.Button.left)
    time.sleep(0.003)

    # Walk along each segment in small steps (generally downward)
    for (sx, sy), (ex, ey) in zip(path[:-1], path[1:]):
        sx, sy, ex, ey = float(sx), float(sy), float(ex), float(ey)
        dx = (ex - sx) / float(steps_per_segment)
        dy = (ey - sy) / float(steps_per_segment)

        x, y = sx, sy
        for _ in range(steps_per_segment):
            x += dx
            y += dy
            mouse_controller.position = (int(x), int(y))
            time.sleep(dt)

    mouse_controller.release(mouse.Button.left)
    time.sleep(0.003)

# --------------
# MAIN LOOP
# --------------

def run_mirror_window(region: Dict[str, int]) -> None:
    """Mirror the selected region, track blobs, highlight upward-moving fruits,
    and perform multi-fruit swipes near the tops of their bounding boxes."""

    last_swipe_frame = -9999

    if not region:
        print("No valid region provided; aborting mirror window.")
        return

    w = region["width"]
    h = region["height"]

    print("Region:", region)
    print("Starting mirror window. Press ESC at any time to quit.\n")

    tracker = Tracker(w, h)

    prev_gray = None
    frame_index = 0

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    frame_area = float(w * h)
    min_side = float(min(w, h))
    swipe_length = max(5, int(SWIPE_LENGTH_RATIO * min_side))

    mouse_controller = mouse.Controller()

    with mss.mss() as sct:
        while True:
            global exit_requested
            if exit_requested:
                print("Exit requested, stopping main loop.")
                break

            screenshot = sct.grab(region)

            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            detections: List[Tuple[float, float, Tuple[int, int, int, int]]] = []

            if prev_gray is not None:
                # --- Frame differencing for motion mask ---
                diff = cv2.absdiff(gray, prev_gray)
                _, motion_mask = cv2.threshold(diff, FRAME_DIFF_THRESH, 255, cv2.THRESH_BINARY)

                # Clean up noise
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

                # Find moving blobs
                contours, _ = cv2.findContours(
                    motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < MIN_AREA_RATIO * frame_area:
                        continue

                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw < MIN_BOX_RATIO * min_side or bh < MIN_BOX_RATIO * min_side:
                        continue

                    cx = x + bw / 2.0
                    cy = y + bh / 2.0

                    detections.append((cx, cy, (x, y, bw, bh)))

                    # GREEN: raw moving blobs for debugging
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

            # Update tracker with detections
            tracker.update(detections, frame_index)

            # Highlight tracks classified as upward-moving fruits
            upward_tracks = tracker.get_upward_fruit_tracks()

            # Collect all current fruit targets (screen coords) for this frame
            current_points: List[Tuple[float, float]] = []

            for tr in upward_tracks:
                x, y, bw, bh = tr.bbox
                cx, cy = tr.positions[-1]

                # RED: fruit candidates (drawing stays the same)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

                label = f"ID {tr.track_id}"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Aim near top of the box (you can keep 0.2 or tweak)
                top_y_local = y + int(0.2 * bh)
                slice_cx = x + bw / 2.0
                slice_cy = top_y_local

                screen_x = region["left"] + slice_cx
                screen_y = region["top"] + slice_cy

                current_points.append((screen_x, screen_y))

            # If we have any fruit this frame and cooldown passed, slice them all
            if current_points and (frame_index - last_swipe_frame) >= SWIPE_COOLDOWN_FRAMES:
                # Sort by something (X for left-right, Y for top-bottom) depending on your swipe style
                # For vertical swipe version, sort by Y:
                current_points.sort(key=lambda p: p[1])

                perform_multi_swipe(
                    mouse_controller,
                    current_points,
                    swipe_length,
                    SWIPE_DURATION_MS,
                    SWIPE_STEPS_PER_SEG,
                )
                last_swipe_frame = frame_index


            # Debug overlay
            debug_text = f"{w}x{h} | tracks={len(tracker.tracks)}"
            cv2.putText(
                frame,
                debug_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Fruit Ninja Tracker View", frame)

            frame_index += 1
            prev_gray = gray

            # 'q' is extra exit; ESC is global
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def main() -> None:
    global exit_requested
    region = select_play_area()

    if not region:
        print("No region selected. Exiting.")
        return

    # Reset exit flag
    exit_requested = False

    # Start global ESC listener (non-blocking)
    esc_listener = keyboard.Listener(on_press=on_global_key_press)
    esc_listener.start()

    try:
        run_mirror_window(region)
    finally:
        esc_listener.stop()
        print("Program terminated.")


if __name__ == "__main__":
    main()
