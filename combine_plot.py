# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc
# —— 新增：对应的移动方差函数 —— #
def _move_var_(in_vec, window_size):
    length = len(in_vec)
    in_vec = np.array(in_vec)
    out_vec = np.zeros(length)
    for i in range(length):
        if i <= window_size - 1:
            window = in_vec[0: i+1]
        else:
            window = in_vec[i - window_size + 1 : i+1]
        out_vec[i] = np.var(window, ddof=0)   # ddof=0 表示总体方差
    return out_vec
def _move_avg_(in_vec, window_size):
    """Compute moving average with fixed window size
        INPUT:
            in_vec: size = (|in_vec|, )
            window_size: scalar, average over past window_size slots
        OUTPUT:
            out_vec: size = (|in_sec|, )
    """
    length =  len(in_vec)
    in_vec = np.array(in_vec)    # convert to np array
    out_vec = np.zeros(length)
    for i in range(length):
        if i <= window_size - 1:
            out_vec[i] = np.mean(in_vec[0: i+1])
        else:
            out_vec[i] = np.mean(in_vec[i - window_size + 1 : i+1])
    return out_vec   




data_ddpg_mul_2 = np.load(f"train_rewards.npy")
#data_ddpg_mul_2_2 = np.load(f"./data/plot_data/channel_1_20_2.npy")
Rwd_over_trials_mul_2 = _move_avg_(data_ddpg_mul_2, window_size=50)


plt.plot (Rwd_over_trials_mul_2, "-",label='Gymnasium DDPG')


plt.xlabel('Number of time steps')
plt.ylabel('Reward')
plt.title('Reward over time steps (Moving Average)')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# =======================
# 2) 画 fake sensor 轨迹
# =======================

# 载入训练时保存的 fake 轨迹 (在 env 里保存的 fake_traj_all.npy)
fake_traj_all = np.load("fake_traj_all.npy")   # shape ~ (n_episodes, T, 2)

print("fake_traj_all shape:", fake_traj_all.shape)

print(fake_traj_all.shape)
# 选择一个 episode 来画：比如最后一个 episode

traj = fake_traj_all[:4000,0,:]   # shape (T, 2)
print(traj.shape)
x_fake = traj[:, 0]
y_fake = traj[:, 1]

# （可选）如果你想同时画出 real sensor 的轨迹：
try:
    real_traj_xy = np.load("real_traj_xy.npy")  # shape (T, d_r, 2)
    # 这里示例只画 real sensor 0 的轨迹
    x_real0 = real_traj_xy[:10000, 0, 0]
    y_real0 = real_traj_xy[:10000, 0, 1]
    x_real1 = real_traj_xy[:10000, 1, 0]
    y_real1 = real_traj_xy[:10000, 1, 1]
    plot_real = True
except FileNotFoundError:
    plot_real = False

plt.figure(figsize=(7, 7))
# fake sensor 轨迹
plt.plot(x_fake, y_fake, '-o', label='Fake sensor traj', alpha=0.8)

# 标出起点 / 终点
plt.scatter(x_fake[0], y_fake[0], c='green', marker='s', s=80, label='Fake start')
plt.scatter(x_fake[-1], y_fake[-1], c='red', marker='x', s=80, label='Fake end')
plt.text(x_fake[0],  y_fake[0],  ' Start', fontsize=10, color='green',
         ha='left', va='bottom')
plt.text(x_fake[-1], y_fake[-1], ' End',   fontsize=10, color='red',
         ha='left', va='bottom')
# 可选：画一个 real sensor 的轨迹对比
if plot_real:
    # 对每个 real sensor 单独画
    colors = ['C1', 'C2', 'C3', 'C4']  # 如果以后 d_r>2 也够用
    for i in range(2):
        x_real = real_traj_xy[:10000, i, 0]
        y_real = real_traj_xy[:10000, i, 1]

        plt.plot(x_real, y_real, '--', alpha=0.7, label=f'Real sensor {i} traj')

        # 起点
        plt.scatter(x_real[0], y_real[0],
                    c=colors[i % len(colors)],
                    marker='^', s=70)
        plt.text(x_real[0], y_real[0],
                 f' Real {i} Start',
                 fontsize=9,
                 color=colors[i % len(colors)],
                 ha='left', va='bottom')

        # 终点
        plt.scatter(x_real[-1], y_real[-1],
                    c=colors[i % len(colors)],
                    marker='v', s=70)
        plt.text(x_real[-1], y_real[-1],
                 f' Real {i} End',
                 fontsize=9,
                 color=colors[i % len(colors)],
                 ha='left', va='bottom')


# 可选：画 jammer 位置
jammer_location = (0.0, 0.0)
plt.scatter([jammer_location[0]], [jammer_location[1]],
            marker='D', s=100, label='Jammer (ULA)', zorder=5)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Fake Sensor Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()