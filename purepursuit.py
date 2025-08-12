import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------- Parameters ----------------------
STRAIGHT_LENGTH = 60.0   # m
RADIUS = 25.0            # m
WHEELBASE = 2.5          # m
DELTA_MAX = np.deg2rad(30.0)  # rad
LOOKAHEAD = 12.0         # m
DT = 0.02                # s
SIM_TIME = 30.0          # s
START_S_FRACTION = 0.05  # along track CCW
V_FIXED = 10.0           # m/s

FPS = 30
FRAME_STRIDE = 2
VIDEO_PATH = "pure_pursuit.mp4"

# ---------------------- Helpers ----------------------
def build_stadium_track(SL: float, R: float, n_per_seg: int = 250):
    cx_right =  SL/2 + R
    cx_left  = -SL/2 - R
    cy = 0.0
    th1 = np.linspace(np.pi/2, 3*np.pi/2, n_per_seg, endpoint=False)
    x1 = cx_right + R*np.cos(th1); y1 = cy + R*np.sin(th1)
    x2 = np.linspace(SL/2, -SL/2, n_per_seg, endpoint=False); y2 = np.full_like(x2, -R)
    th3 = np.linspace(3*np.pi/2, 5*np.pi/2, n_per_seg, endpoint=False)
    x3 = cx_left + R*np.cos(th3); y3 = cy + R*np.sin(th3)
    x4 = np.linspace(-SL/2, SL/2, n_per_seg, endpoint=False); y4 = np.full_like(x4, R)
    x_ref = np.concatenate([x1, x2, x3, x4])
    y_ref = np.concatenate([y1, y2, y3, y4])
    return x_ref, y_ref

def cumulative_arclength(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0.0)[:-1]
    return s

def nearest_index_windowed(x_ref, y_ref, x, y, prev_idx, window=30):
    n = len(x_ref)
    idxs = (np.arange(prev_idx - window, prev_idx + window + 1) % n).astype(int)
    d = (x_ref[idxs] - x)**2 + (y_ref[idxs] - y)**2
    k = int(np.argmin(d))
    return int(idxs[k])

def lookahead_index_from(i_start, x_ref, y_ref, Ld):
    n = len(x_ref)
    acc = 0.0; i = i_start
    while acc < Ld:
        j = (i + 1) % n
        acc += np.hypot(x_ref[j] - x_ref[i], y_ref[j] - y_ref[i])
        i = j
        if i == i_start: break
    return i

def pure_pursuit_delta(x, y, yaw, x_ref, y_ref, Ld, L, delta_max, idx_hint):
    idx_near = nearest_index_windowed(x_ref, y_ref, x, y, idx_hint, window=30)
    idx_tgt = lookahead_index_from(idx_near, x_ref, y_ref, Ld)
    dx = x_ref[idx_tgt] - x; dy = y_ref[idx_tgt] - y
    c, s = np.cos(-yaw), np.sin(-yaw)
    xf = c*dx - s*dy; yf = s*dx + c*dy
    Ld_eff = max(1e-6, np.hypot(xf, yf))
    curvature = 2.0 * yf / (Ld_eff**2)
    delta = np.arctan(L * curvature)
    return float(np.clip(delta, -delta_max, +delta_max)), idx_near, idx_tgt

# ---------------------- Simulate ----------------------
x_ref, y_ref = build_stadium_track(STRAIGHT_LENGTH, RADIUS, n_per_seg=220)
s_ref = cumulative_arclength(x_ref, y_ref)
s_total = s_ref[-1] + np.hypot(x_ref[-1]-x_ref[0], y_ref[-1]-y_ref[0])

s_start = (START_S_FRACTION % 1.0) * s_total
idx0 = int(np.searchsorted(s_ref, s_start, side="left"))
x = float(x_ref[idx0]); y = float(y_ref[idx0])
j = (idx0 + 1) % len(x_ref)
yaw = np.arctan2(y_ref[j] - y_ref[idx0], x_ref[j] - x_ref[idx0])

steps = int(SIM_TIME / DT)
xs = np.empty(steps); ys = np.empty(steps); yaws = np.empty(steps)
idx_tgts = np.empty(steps, dtype=int)
idx_hint = idx0

for k in range(steps):
    delta, idx_near, idx_tgt = pure_pursuit_delta(x, y, yaw, x_ref, y_ref, LOOKAHEAD, WHEELBASE, DELTA_MAX, idx_hint)
    v = V_FIXED
    x += v * np.cos(yaw) * DT
    y += v * np.sin(yaw) * DT
    yaw += (v / WHEELBASE) * np.tan(delta) * DT
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi
    xs[k] = x; ys[k] = y; yaws[k] = yaw; idx_tgts[k] = idx_tgt
    idx_hint = idx_near

# ---------------------- Animation ----------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_ref, y_ref, linewidth=1.5, label="Centerline")
traj_line, = ax.plot([], [], linewidth=1.2, label="Vehicle path")
car_point, = ax.plot([], [], marker='o', markersize=6, linestyle='None', label="Vehicle")
look_point, = ax.plot([], [], marker='x', markersize=7, linestyle='None', label="Look-ahead")
heading_quiver = ax.quiver([], [], [], [])

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Pure Pursuit (fixed speed) — vehicle, heading, and look-ahead")
ax.legend(loc="best")
ax.set_xlim(x_ref.min()-10, x_ref.max()+10)
ax.set_ylim(y_ref.min()-10, y_ref.max()+10)
arrow_len = 4.0

frames = range(0, steps, FRAME_STRIDE)

def init():
    traj_line.set_data([], [])
    car_point.set_data([], [])
    look_point.set_data([], [])
    return traj_line, car_point, look_point, heading_quiver

def update(frame_idx):
    i = frame_idx
    traj_line.set_data(xs[:i+1], ys[:i+1])
    car_point.set_data(xs[i], ys[i])
    u = arrow_len * np.cos(yaws[i]); v = arrow_len * np.sin(yaws[i])
    ax.collections = [c for c in ax.collections if c not in [heading_quiver]]
    heading = ax.quiver(xs[i], ys[i], u, v, angles='xy', scale_units='xy', scale=1)
    j = idx_tgts[i]
    look_point.set_data(x_ref[j], y_ref[j])
    return traj_line, car_point, look_point, heading

ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                              blit=False, interval=1000/FPS)

# Save MP4
# try:
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=FPS)
ani.save(VIDEO_PATH, writer=writer)
print(f"Saved video to: {VIDEO_PATH}")
# except Exception as e:
#     print(f"Could not save MP4: {e}")

VIDEO_PATH
