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
DY_UP_THRESH_NORM = 0.008      # mean(dy_norm) must be < -DY_UP_THRESH_NORM to be "upward"
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

# -------------------------
# TRACKING DATA STRUCTURES
# -------------------------

@dataclass
class Track:
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0,0,0,0)
    last_seen_frame: int = 0

    def add_detection(self, cx, cy, bbox, frame_idx):
        self.positions.append((cx, cy))
        if len(self.positions) > MAX_HISTORY_LEN:
            self.positions.pop(0)
        self.bbox = bbox
        self.last_seen_frame = frame_idx

    def is_upward_fruit(self, frame_w, frame_h):
        """Classify as upward-moving fruit."""
        if len(self.positions) < MIN_HISTORY_FOR_CLASS:
            return False

        k = min(MIN_HISTORY_FOR_CLASS, len(self.positions) - 1)
        pts = self.positions[-(k+1):]

        dys = []
        dxs = []
        for (x1,y1),(x2,y2) in zip(pts[:-1], pts[1:]):
            dxs.append(x2 - x1)
            dys.append(y2 - y1)

        if not dys:
            return False

        dy_norm = np.array(dys) / float(frame_h)
        dx_norm = np.array(dxs) / float(frame_w)

        mean_dy = float(np.mean(dy_norm))
        mean_dx = float(np.mean(np.abs(dx_norm)))

        cx, cy = self.positions[-1]

        # bottom window constraint
        if not (BOTTOM_REGION_MIN_RATIO * frame_h <= cy <= BOTTOM_REGION_MAX_RATIO * frame_h):
            return False

        # upward motion
        if mean_dy >= -DY_UP_THRESH_NORM:
            return False

        # mostly vertical
        if mean_dx > DX_MAX_FRUIT_NORM:
            return False

        # last step must not be downward
        if dy_norm[-1] > 0:
            return False

        return True


class Tracker:
    def __init__(self, w, h):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_w = w
        self.frame_h = h
        self.area = w*h
        self.diag = float(math.hypot(w,h))
        self.min_side = float(min(w,h))

    def _dist(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def update(self, detections, frame_idx):
        """detections = [(cx,cy,bbox), ...]"""
        unmatched_det = set(range(len(detections)))
        unmatched_trk = set(self.tracks.keys())

        maxd = MAX_ASSOC_DIST_RATIO * self.diag
        pairs = []

        for tid,tr in self.tracks.items():
            if not tr.positions:
                continue
            tx,ty = tr.positions[-1]
            for i,(cx,cy,_) in enumerate(detections):
                d = self._dist((tx,ty),(cx,cy))
                if d <= maxd:
                    pairs.append((d, tid, i))

        pairs.sort(key=lambda x:x[0])

        # assign pairs
        for d,tid,i in pairs:
            if tid not in unmatched_trk:
                continue
            if i not in unmatched_det:
                continue
            unmatched_trk.remove(tid)
            unmatched_det.remove(i)

            cx,cy,bbox = detections[i]
            self.tracks[tid].add_detection(cx,cy,bbox,frame_idx)

        # create new tracks
        for i in unmatched_det:
            cx,cy,bbox = detections[i]
            tr = Track(self.next_id)
            tr.add_detection(cx,cy,bbox,frame_idx)
            self.tracks[self.next_id] = tr
            self.next_id += 1

        # prune stale
        kill = [tid for tid,tr in self.tracks.items()
                if frame_idx - tr.last_seen_frame > MAX_MISSED_FRAMES]
        for tid in kill:
            del self.tracks[tid]

    def get_upward_fruit_tracks(self):
        return [tr for tr in self.tracks.values()
                if tr.is_upward_fruit(self.frame_w, self.frame_h)]


# -------------------------
# MOUSE SWIPE
# -------------------------

def perform_multi_swipe(mouse_ctl, points, swipe_length, duration_ms, steps_per_seg):
    """
    Vertical top→bottom swipe through all points.
    """
    if not points:
        return

    # Sort by Y (top to bottom)
    points = sorted(points, key=lambda p:p[1])

    first_x, first_y = points[0]
    last_x,  last_y  = points[-1]

    L = float(swipe_length)

    start = (first_x, first_y - L/2)
    end   = (last_x,  last_y + L/2)

    path = [start] + points + [end]

    if len(path) < 2:
        return

    if steps_per_seg <= 0:
        steps_per_seg = 1

    total_steps = max(1, (len(path)-1)*steps_per_seg)
    dt = (duration_ms/1000.0)/float(total_steps)

    x0,y0 = path[0]
    mouse_ctl.position = (int(x0),int(y0))
    time.sleep(0.003)
    mouse_ctl.press(mouse.Button.left)
    time.sleep(0.003)

    # interpolate each segment
    for (sx,sy),(ex,ey) in zip(path[:-1], path[1:]):
        sx,sy,ex,ey = map(float,(sx,sy,ex,ey))
        dx = (ex-sx)/steps_per_seg
        dy = (ey-sy)/steps_per_seg

        x,y = sx,sy
        for _ in range(steps_per_seg):
            x += dx
            y += dy
            mouse_ctl.position = (int(x),int(y))
            time.sleep(dt)

    mouse_ctl.release(mouse.Button.left)
    time.sleep(0.003)


# -------------------------
# GLOBAL ESC HANDLER
# -------------------------

def on_global_key_press(key):
    global exit_requested
    if key == keyboard.Key.esc:
        exit_requested = True
        return False


# -------------------------
# REGION SELECTION
# -------------------------

def select_play_area():
    pts = []
    mouse_ctl = mouse.Controller()

    print("\n=== Configure Play Area ===")
    print("Hover over FIRST corner and press SPACE.")
    print("Hover over OPPOSITE corner and press SPACE again.")
    print("Press ESC to cancel.\n")

    def handler(k):
        nonlocal pts
        if k == CANCEL_KEY:
            print("Cancelled.")
            return False
        if k == CONFIRM_KEY:
            x,y = mouse_ctl.position
            pts.append((x,y))
            print(f"Captured point {len(pts)}: {x},{y}")
            if len(pts)==2:
                return False

    with keyboard.Listener(on_press=handler) as ls:
        ls.join()

    if len(pts)!=2:
        return {}

    (x1,y1),(x2,y2) = pts
    left   = min(x1,x2)
    top    = min(y1,y2)
    right  = max(x1,x2)
    bottom = max(y1,y2)

    return {
        "left": left,
        "top": top,
        "width": right-left,
        "height": bottom-top
    }


# -------------------------
# MAIN LOOP
# -------------------------

def run_mirror_window(region):

    if not region:
        print("No region.")
        return

    w = region["width"]
    h = region["height"]

    tracker = Tracker(w,h)

    prev_gray = None
    frame_idx = 0

    mouse_ctl = mouse.Controller()
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    frame_area = float(w*h)
    min_side = float(min(w,h))
    swipe_length = max(5, int(SWIPE_LENGTH_RATIO * min_side))

    last_swipe_frame = -9999  # cooldown tracker

    with mss.mss() as sct:
        while True:
            global exit_requested
            if exit_requested:
                break

            screenshot = sct.grab(region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # -------------------------
            # BOMB DETECTION
            # -------------------------
            # Pure red: R = 255→~156, G=B=0
            lower_bomb = np.array([0,0,156], dtype=np.uint8)
            upper_bomb = np.array([20,20,255], dtype=np.uint8)

            bomb_mask = cv2.inRange(frame, lower_bomb, upper_bomb)
            bomb_mask = cv2.morphologyEx(bomb_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            bomb_mask = cv2.morphologyEx(bomb_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

            bomb_ctrs,_ = cv2.findContours(bomb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bomb_boxes = []
            TOP_IGNORE_Y = int(0.10 * h)   # IGNORE top 10% of play area

            for c in bomb_ctrs:
                if cv2.contourArea(c) < 80:
                    continue

                bx,by,bw,bh = cv2.boundingRect(c)

                # ----------------------------
                # IGNORE BOMBS IN TOP 10% (these are icons, not real bombs)
                # ----------------------------
                if by < TOP_IGNORE_Y:
                    continue

                bomb_boxes.append((bx,by,bw,bh))

                # (Optional visualization — remove if you like)
                cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(255,0,0),2)
                cv2.putText(frame,"BOMB",(bx,by-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)


            # -------------------------
            # MOTION DETECTION
            # -------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(5,5),0)

            detections = []

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _,mask = cv2.threshold(diff, FRAME_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < MIN_AREA_RATIO * frame_area:
                        continue
                    x,y,bw,bh = cv2.boundingRect(c)
                    if bw < MIN_BOX_RATIO * min_side or bh < MIN_BOX_RATIO * min_side:
                        continue

                    cx = x + bw/2
                    cy = y + bh/2
                    detections.append((cx,cy,(x,y,bw,bh)))

                    # green box
                    cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),1)

            # -------------------------
            # TRACKING + FRUIT CLASSIFICATION
            # -------------------------
            tracker.update(detections, frame_idx)
            red_tracks = tracker.get_upward_fruit_tracks()

            # -------------------------
            # COLLECT SAFE FRUIT TARGETS (bomb exclusion)
            # -------------------------
            safe_points = []

            for tr in red_tracks:
                x,y,bw,bh = tr.bbox
                cx,cy = tr.positions[-1]

                # draw red box
                cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,0,255),2)
                cv2.circle(frame, (int(cx),int(cy)), 4, (0,0,255), -1)

                # compute slice point near top of bbox
                sy_local = y + int(0.2*bh)
                slice_cx = region["left"] + (x + bw/2.0)
                slice_cy = region["top"]  + sy_local

                # BOMB AVOIDANCE: skip if point lies inside any bomb bounding box
                in_bomb = False
                for (bx,by,bw2,bh2) in bomb_boxes:
                    if bx <= (x+bw/2) <= (bx+bw2) and by <= sy_local <= (by+bh2):
                        in_bomb = True
                        break

                if not in_bomb:
                    safe_points.append((slice_cx, slice_cy))

            # -------------------------
            # SWIPE (VERTICAL) IF SAFE FRUITS EXIST
            # -------------------------
            if safe_points and (frame_idx - last_swipe_frame >= SWIPE_COOLDOWN_FRAMES):
                perform_multi_swipe(
                    mouse_ctl,
                    safe_points,
                    swipe_length,
                    SWIPE_DURATION_MS,
                    SWIPE_STEPS_PER_SEG
                )
                last_swipe_frame = frame_idx

            # -------------------------
            # DEBUG TEXT
            # -------------------------
            dbg = f"{w}x{h}  Tracks:{len(tracker.tracks)}  Bombs:{len(bomb_boxes)}"
            cv2.putText(frame, dbg, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

            cv2.imshow("NinjaBot View", frame)

            prev_gray = gray
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


# -------------------------
# MAIN
# -------------------------

def main():
    global exit_requested
    region = select_play_area()
    if not region:
        print("No region selected.")
        return

    exit_requested = False
    esc_listener = keyboard.Listener(on_press=on_global_key_press)
    esc_listener.start()

    try:
        run_mirror_window(region)
    finally:
        esc_listener.stop()
        print("Program terminated.")


if __name__ == "__main__":
    main()