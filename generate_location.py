import numpy as np
import matplotlib.pyplot as plt
import random
M = 4  # Number of antennas at the array (Uniform Linear Array, ULA)
d_r = 2  # Number of real sources (true sensor nodes)
d_f = 1  # Number of fake sources (decoys we will optimize)
real_aoas = [20, 60]  # True AoAs (in degrees) of the real sources
sigma2 = 1e-3  # Noise power (variance of additive white Gaussian noise)
theta_grid = np.linspace(-90, 90, 181)  # Grid of scanning angles (-90° to 90°, 0.1° resolution)
d_over_lambda = 0.5  # Antenna spacing normalized to wavelength (half-wavelength spacing)
d_total = d_r + d_f  # Total number of sources MUSIC will detect (important!)

# generate location for jammer (global coords)
jammer_location = (0.0, 0.0)   # ULA center location（GLOBAL）

# generate location for real and fake nodes (GLOBAL coords)
K = d_r + d_f
Lroom = 10.0   # radius x
Wroom = 20.0   # radius y
radius = 10.0       # radius distance (m)

nodes_xy = np.zeros((K, 2), dtype=float)
center = np.array([Lroom, Wroom], dtype=float)

rng = np.random.default_rng()  # can add seed rng = np.random.default_rng(123)

for k0 in range(K):
    # uniform sample in the darius：r = R * sqrt(U), angle ~ U[0, 2π]
    r = radius * np.sqrt(rng.uniform(0.0, 1.0))
    angle = rng.uniform(0.0, 2*np.pi)
    offset = np.array([r*np.cos(angle), r*np.sin(angle)], dtype=float)
    pt = center + offset  # GLOBAL coords
    nodes_xy[k0, :] = pt


print("Node Locations (GLOBAL, x,y):\n", nodes_xy)
fig, ax = plt.subplots(figsize=(7, 7))

# (x,y) 
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)

# ULA location
ax.scatter([jammer_location[0]], [jammer_location[1]],
           marker='s', s=100, label='ULA (jammer center)')

# real / fake node ( nodes_xy ：previous 2 d_r is real，last one d_f is fake）
real_location = nodes_xy[:d_r, :]
fake_location = nodes_xy[d_r:, :]

ax.scatter(real_location[:, 0], real_location[:, 1],
           marker='o', s=60, label=f'Real (n={d_r})')
ax.scatter(fake_location[:, 0], fake_location[:, 1],
           marker='x', s=80, label=f'Fake (n={d_f})')

# radius bound（center (Lroom, Wroom)， R is distance from center）
theta = np.linspace(0, 2*np.pi, 256)
circle_x = Lroom + radius * np.cos(theta)
circle_y = Wroom + radius * np.sin(theta)
ax.plot(circle_x, circle_y, linewidth=1.0, label='Sampling circle')
ax.scatter([Lroom], [Wroom], marker='+', s=80, label='Circle center')

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Global frame: ULA (jammer), real/fake nodes and sampling circle')
ax.grid(True)
ax.legend(loc='best')
plt.tight_layout()
plt.show()

np.save("real_sensor_location.npy", real_location)
np.save("fake_sensor_location.npy", fake_location)  

T= 20000
dt = 1
v_max = 0.001
speeds = rng.uniform(0.001, v_max, size=K)          # 每个节点一个速度标量
directions = rng.uniform(0, 2*np.pi, size=K)    # 每个节点的运动方向
vel_xy = np.stack([speeds * np.cos(directions),
                   speeds * np.sin(directions)], axis=1)  # shape = (K, 2)

# 生成轨迹: positions[t, k, :] 表示 t 时刻第 k 个节点的 (x,y)
traj_xy = np.zeros((T, K, 2), dtype=float)
for t in range(T):
    traj_xy[t, :, :] = nodes_xy + vel_xy * (t * dt)

# 拆成 real / fake 的轨迹
real_traj_xy = traj_xy[:, :d_r, :]   # shape = (T, d_r, 2)
fake_traj_xy = traj_xy[:, d_r:, :]   # shape = (T, d_f, 2)

print("real_traj_xy shape:", real_traj_xy.shape)
print("fake_traj_xy shape:", fake_traj_xy.shape)

# 1) global cordinates
ARRAY_CENTER_GLOBAL = np.array(jammer_location, dtype=float)
ARRAY_YAW_DEG = 0.0  # If ULA is not globle, shift to global

def to_array_frame(points_xy, array_center=ARRAY_CENTER_GLOBAL, yaw_deg=ARRAY_YAW_DEG):
    pts = np.asarray(points_xy, dtype=float)                  # Nx2
    pts_shift = pts - np.asarray(array_center)[None, :]       # Shift
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(-yaw), np.sin(-yaw)                         # Rotation: Align ULA to the x-axis
    Rm = np.array([[c, -s], [s,  c]])
    return (pts_shift @ Rm.T)                                  # Return (x', y') in the array coordinate frame

# 2) Convert real-node positions from global to array coordinate frame
real_xy_array = to_array_frame(real_location)  # shape: (d_r, 2)

# 3) AoA computation: θ = atan2(x', y') (note: use atan2(x', y'), broadside = +y)
#real_aoas = list(np.degrees(np.arctan2(real_xy_array[:, 0], real_xy_array[:, 1])))
"This can also be directly set, e.g., real_aoas = [20, 60]"
# 4) Convert fake-node positions to the array frame and compute distances from the array for real/fake
fake_xy_array = to_array_frame(fake_location)  # shape: (d_f, 2)

# print("Real locations (ARRAY frame):\n", real_xy_array)
# print("Real AoAs (deg) w.r.t. jammer ULA:", real_aoas)
jammer_x, jammer_y = jammer_location

# # Direct distance of real nodes (to the jammer/ULA phase center)
# real_dist = np.hypot(real_location[:, 0] - jammer_x,
#                      real_location[:, 1] - jammer_y)      # shape: (d_r,)

# # Direct distance of fake nodes (d_f = 1)
# fake_dist = np.hypot(fake_location[:, 0] - jammer_x,
#                          fake_location[:, 1] - jammer_y)   # shape: (d_f,)

# 距离: dist[t, i] = t 时刻第 i 个 real 到 jammer 的距离
real_dist_t = np.zeros((T, d_r), dtype=float)
fake_dist_t = np.zeros((T, d_f), dtype=float)

for t in range(T):
    # t 时刻 real 的位置
    real_xy_t = real_traj_xy[t, :, :]   # shape (d_r, 2)
    fake_xy_t = fake_traj_xy[t, :, :]   # shape (d_f, 2)

    real_dist_t[t, :] = np.hypot(real_xy_t[:, 0] - jammer_x,
                                 real_xy_t[:, 1] - jammer_y)
    fake_dist_t[t, :] = np.hypot(fake_xy_t[:, 0] - jammer_x,
                                 fake_xy_t[:, 1] - jammer_y)

print(real_dist_t.shape)
print(fake_dist_t.shape)
np.save("real_sensor_dist.npy", real_dist_t)
np.save("fake_sensor_dist.npy", fake_dist_t)  
np.save("real_traj_xy.npy", real_traj_xy)
np.save("fake_traj_xy.npy", fake_traj_xy)  

T = real_traj_xy.shape[0]

plt.figure(figsize=(7, 7))

# 画 jammer 位置
plt.scatter([jammer_location[0]], [jammer_location[1]],
            marker='s', s=120, label='ULA (jammer)', zorder=5)

# 画 real 节点轨迹
for i in range(d_r):
    x = real_traj_xy[:, i, 0]
    y = real_traj_xy[:, i, 1]

    # 轨迹线
    plt.plot(x, y, '-', alpha=0.8, label=f'Real {i} traj')

    # 起点
    plt.scatter(x[0], y[0], c='C0', marker='o', s=60, zorder=6)
    plt.text(x[0], y[0], ' Start', fontsize=9, ha='left', va='bottom')

    # 终点
    plt.scatter(x[-1], y[-1], c='C0', marker='x', s=80, zorder=6)
    plt.text(x[-1], y[-1], ' End', fontsize=9, ha='left', va='bottom')


plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Real / Fake Trajectories (Start & End Marked)')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()