import numpy as np
import matplotlib.pyplot as plt
import random

# --- System Parameters ---
# --- 加载 Gauss–Markov 信道 ---
channel_file = "gauss_markov_channel_rho0.9.npy"
G_sig = np.load(channel_file)  # shape = (N, M, ite)
real_location = np.load("real_sensor_location.npy")  # shape = (d_r, 2)
fake_location = np.load("fake_sensor_location.npy")  # shape = (d_f, 2)
real_traj_xy = np.load("real_traj_xy.npy")  # shape = (T, d_r, 2)
real_dist_t = np.load("real_sensor_dist.npy")  # shape = (T, d_r)
T = real_traj_xy.shape[0]  # Number of time steps
# generate location for jammer (global coords)
jammer_location = (0.0, 0.0)   # ULA center location（GLOBAL）
# generate location for real and fake nodes (GLOBAL coords)
M = 4  # Number of antennas at the array (Uniform Linear Array, ULA)
d_r = 2  # Number of real sources (true sensor nodes)
d_f = 1  # Number of fake sources (decoys we will optimize)
K = d_r + d_f
#real_aoas = [20, 60]  # True AoAs (in degrees) of the real sources
real_aoas_t = []
print("real_traj_xy shape:", real_traj_xy.shape)
for i in range(T):
    real_xy_t = real_traj_xy[i, :, :]   # shape (d_r, 2)
    dx = real_xy_t[:, 0] - jammer_location[0]  # jammer_x = 0.0
    dy = real_xy_t[:, 1] - jammer_location[1]  # jammer_y = 0.0
    real_aoas_t.append(np.degrees(np.arctan2(dx, dy)))  # shape (d_r,)

sigma2 = 1e-3  # Noise power (variance of additive white Gaussian noise)
theta_grid = np.linspace(-90, 90, 181)  # Grid of scanning angles (-90° to 90°, 0.1° resolution)
d_over_lambda = 0.5  # Antenna spacing normalized to wavelength (half-wavelength spacing)
d_total = d_r + d_f  # Total number of sources MUSIC will detect (important!)

real_power = 1.0  # Power of each real source
fake_power = 1.0  # Power of each fake source
"get every time step real sensors effective power (pathloss 1/(d^2))"
real_Peff_t = (real_power * np.ones_like(real_dist_t)) / np.maximum(real_dist_t, 1e-12)**2
"create the empty fake sensor power array"
fake_Peff_t = np.zeros(real_Peff_t.shape)
#print(fake_Peff_t[1,:])
#fake_Peff_t = (fake_power * np.ones_like(fake_dist_t)) / np.maximum(fake_dist_t, 1e-12)**2


"----------------------------------------------------------------------------"
# --- Helper Functions ---
def steering_vector(theta, M, d_over_lambda=0.5):
    """Compute the array steering vector for a given angle theta."""
    m = np.arange(M)  # Antenna element indices: 0,1,...,M-1
    return np.exp(-1j * 2 * np.pi * d_over_lambda * m * np.sin(np.radians(theta)))

def generate_perturbation_matrix(M):
    """Generate a small Hermitian complex matrix E to perturb R."""
    A = (np.random.randn(M, M) + 1j * np.random.randn(M, M)) / np.sqrt(2)
    scale = 0 #1e-2
    E = scale * (A + A.conj().T) / 2  # Make it Hermitian
    return E
def compute_covariance_matrix(real_aoas, fake_aoas, M, sigma2, 
                              real_power, fake_power,
                              h_real_t, h_fake_t,
                              ):
    """
    Build the array covariance matrix:
    R = sum over real sources + sum over fake sources + noise.
    Each source contributes a rank-1 matrix scaled by its power.
    """
   
    all_aoas = list(real_aoas) + list(fake_aoas)

    #all_powers = [real_power] + [fake_power]
    all_powers = np.concatenate([real_power, fake_power])
    all_powers = all_powers.tolist()
    all_h = np.concatenate([h_real_t, h_fake_t]) 
 
    R = sigma2 * np.eye(M, dtype=complex)  # Start with noise
    "R的维度是1x4"
    # for theta, power in zip(all_aoas, all_powers):
    #     a = steering_vector(theta, M)
    #     R += power * np.outer(a, np.conj(a))  # Add rank-1 contribution for each source

    # Add perturbation error
    # 这里利用全局变量 G_snapshot: shape (K, M)
    # for src_idx, (theta, power) in enumerate(zip(all_aoas, all_powers)):
    #     a_theta = steering_vector(theta, M)      # 纯几何 steering
    #     h_vec = G_snapshot[src_idx, :]          # 这一源在 t=0 的阵列信道向量, shape (M,)
    #     a_eff = h_vec * a_theta                  # 逐元素加 channel

    #     R += power * np.outer(a_eff, np.conj(a_eff))

    for theta, power, hk in zip(all_aoas, all_powers, all_h):
        a = steering_vector(theta, M)

        v = hk * a   # hk 是标量，只改变幅度/整体相位
        R += power * np.outer(v, np.conj(v))
    Error = generate_perturbation_matrix(M)  # Adjust scale if needed
    R = R + Error

    return R


def compute_music_spectrum(R, M, d_total, theta_grid):
    """
    Compute MUSIC pseudo-spectrum over all scanning angles.
    Peaks occur where steering vector aligns with signal subspace.
    """
    eigvals, eigvecs = np.linalg.eigh(R)  # Eigen-decomposition of covariance matrix
    Un = eigvecs[:, :-d_total]  # Noise subspace (smallest M-d_total eigenvectors)

    spectrum = []
    for theta in theta_grid:
        a_theta = steering_vector(theta, M)
        denom = np.linalg.norm(Un.conj().T @ a_theta) ** 2  # Projection onto noise subspace
        spectrum.append(1.0 / denom)  # MUSIC pseudo-spectrum value
    return np.array(spectrum)

def compute_entropy(p):
    """
    Compute Shannon entropy of the normalized MUSIC spectrum:
    Higher entropy = flatter, more uniform spectrum = more confusion.
    """
    p_safe = np.clip(p, 1e-12, None)  # Avoid log(0)
    return -np.sum(p_safe * np.log2(p_safe))  # Compute entropy in bits

def _window_entropy_at_theta(music_norm, theta_grid, theta, half_win=1):
    idx = int(np.argmin(np.abs(theta_grid - theta)))
    lo = max(0, idx - half_win)
    hi = min(len(theta_grid) - 1, idx + half_win)
    p = music_norm[lo:hi+1].astype(float)
    p = p / max(p.sum(), 1e-12)
    p = np.clip(p, 1e-12, None)
    return -np.sum(p * np.log2(p))

# ================================
# === ADD-ON: Gym Environment  ===
# ================================
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    print("Gymnasium not installed.")
    raise

class FakeLocationEnv(gym.Env):
    """
    state = [ real_x1, real_y1, ..., real_x_d_r, real_y_d_r,
              Re(h_real_1), Im(h_real_1), ..., Re(h_real_d_r), Im(h_real_d_r) ]
    action = [x_fake, y_fake]
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 G_sig,
                 real_traj_xy,
                 real_Peff_t,
                 theta_grid,
                 M, d_total, sigma2,
                 real_aoas,
                 jammer_location,
                 fake_power=1.0,
                 lambda_weight_fake=0.1,
                 entropy_halfwin=1,
                 fake_init_xy=fake_location,          # ★ 初始 fake 位置
                 max_delta_pos=0.001,          # ★ 每一步最大位移（m）
                 ):

        super().__init__()
        self.G_sig = G_sig
        self.real_traj_xy = real_traj_xy

        self.real_Peff_t = real_Peff_t
        self.theta_grid = np.asarray(theta_grid)
        self.M = M
        self.d_total = d_total
        self.sigma2 = sigma2
        self.real_aoas = real_aoas_t
        self.d_r = real_traj_xy.shape[1]
        self.d_f = 1
        self.K, _, self.T = G_sig.shape
        self.jammer_location = np.asarray(jammer_location, dtype=float)
        self.fake_power = float(fake_power)
        self.lambda_weight_fake = float(lambda_weight_fake)
        self.entropy_halfwin = int(entropy_halfwin)
        self.fake_xy0 = np.asarray(fake_init_xy, dtype=float)

        # ---- 动作空间: [Δx, Δy] ----
        self.max_delta_pos = float(max_delta_pos)
        # ★ 动作空间：Δx, Δy, Δθ

        self.action_space = spaces.Box(
            low=np.array([-self.max_delta_pos, -self.max_delta_pos], dtype=np.float32),
            high=np.array([self.max_delta_pos, self.max_delta_pos], dtype=np.float32),
            dtype=np.float32
        )

        # ---- observation space bounds ----
        # loc 部分用上面的范围；channel 部分用 G_sig 的最大幅度估计
        h_max = float(np.max(np.abs(G_sig))) + 1.0
        obs_dim = 4 * self.d_r
        x_min = np.min(real_traj_xy[:, :, 0]) - 10.0
        x_max = np.max(real_traj_xy[:, :, 0]) + 10.0
        y_min = np.min(real_traj_xy[:, :, 1]) - 10.0
        y_max = np.max(real_traj_xy[:, :, 1]) + 10.0
        low_loc = np.array([x_min, y_min] * self.d_r, dtype=np.float32)
        high_loc = np.array([x_max, y_max] * self.d_r, dtype=np.float32)

        low_h = -h_max * np.ones(2 * self.d_r, dtype=np.float32)
        high_h = h_max * np.ones(2 * self.d_r, dtype=np.float32)
         # ★ 这里把 “上一时刻的 action” 也并到 observation 范围里
        low_obs = np.concatenate([low_loc, low_h, self.action_space.low])
        high_obs = np.concatenate([high_loc, high_h, self.action_space.high])
        self.observation_space = spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32
        )

        self.t = 0  # current time step
        self.reward = []
        #
        # 维护 fake 当前坐标 + 上一时刻动作
        self.fake_xy = self.fake_xy0.copy()

        self.last_action = np.zeros(2, dtype=np.float32)   # [Δx_prev, Δy_prev]
        # ★ 用来记录轨迹
        self.fake_traj_current = []   # 当前 episode 的轨迹（列表，每个是 [x,y]）
        self.fake_traj_all = []       # 所有 episode 的轨迹（列表，每个是 (T_i,2) array）
    # ---- 构造当前 state ----
    def _get_state(self):
        real_xy_t = self.real_traj_xy[self.t]            # (d_r, 2)
        h_real_t = self.G_sig[:self.d_r, 0, self.t]      # (d_r,)

        loc_flat = real_xy_t.reshape(-1)  # 2*d_r
        h_stack = np.stack([h_real_t.real, h_real_t.imag], axis=-1).reshape(-1)  # 2*d_r

        state = np.concatenate([loc_flat, h_stack, self.last_action]).astype(np.float32)
        return state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.fake_xy = self.fake_xy0.copy()
        self.last_action = np.zeros(2, dtype=np.float32)
        self.rewards = []
        # ★ 重置当前 episode 轨迹，并记录初始点
        self.fake_traj_current = [self.fake_xy.copy()]
        obs = self._get_state()
        return obs, {}


    def step(self, action):
        # clip action into valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy = float(action[0]), float(action[1])

        # ★ 先更新 fake 位置 & 角度偏移
        # 更新 fake 位置
   
        self.fake_xy += np.array([dx, dy], dtype=float)
        # ★ 把新位置加入当前轨迹
        self.fake_traj_current.append(self.fake_xy.copy())

        dx_f = self.fake_xy[0,0] - self.jammer_location[0]
        dy_f = self.fake_xy[0,1] - self.jammer_location[1]
        theta_fake = np.degrees(np.arctan2(dx_f, dy_f))
        theta_fake = float(np.clip(theta_fake, -90.0, 90.0))
        fake_aoas = [theta_fake]
        
        real_aoas = np.array(self.real_aoas[self.t])  # shape (d_r,)
        #print(real_aoas)
        dist_f = np.sqrt(dx_f**2 + dy_f**2)
        fake_Peff_vec = np.array(
            [self.fake_power / max(dist_f**2, 1e-12)],
            dtype=float
        )  # shape (1,)

        # current real Peff and channels
        real_Peff_vec = self.real_Peff_t[self.t, :]           # (d_r,)
        h_real_t = self.G_sig[:self.d_r, 0, self.t]           # (d_r,)
        h_fake_t = self.G_sig[self.d_r:, 0, self.t]           # (d_f,)

        # ---- covariance & MUSIC ----
        R = compute_covariance_matrix(
            real_aoas, fake_aoas,
            self.M, self.sigma2,
            real_Peff_vec, fake_Peff_vec,
            h_real_t,h_fake_t
        )
        music = compute_music_spectrum(R, self.M, self.d_total, self.theta_grid)
        
        music_norm = music / np.sum(music)
        #entropy = compute_entropy(music_norm)

        # ---- real peak sum ----
        real_peak_sum = 0.0
        for th in real_aoas:
            idx = np.argmin(np.abs(self.theta_grid - th))
            real_peak_sum += float(music_norm[idx])

        # # ---- entropy around fake angle ----
        # entropy_fake = _window_entropy_at_theta(
        #     music_norm, self.theta_grid, theta_fake,
        #     half_win=self.entropy_halfwin
        # )

        fake_values = []
        for th in fake_aoas:
            idx = np.argmin(np.abs(self.theta_grid - th))
            fake_values.append(music_norm[idx])
        fake_values = np.array(fake_values)
        fake_values /= np.sum(fake_values)
  
        entropy_fake = compute_entropy(fake_values)
     
        objective = real_peak_sum - self.lambda_weight_fake * entropy_fake
        reward = -objective
        self.reward.append(reward)
        print(reward)
         # 记录本步动作到 last_action
        self.last_action = np.array([dx, dy], dtype=np.float32)
        # ---- next state / termination ----
        self.t += 1
        terminated = self.t >= (self.T - 1)
        truncated = False

        obs = self._get_state()
        info = dict(
            t=self.t,
            fake_xy=self.fake_xy.copy(),
            fake_aoa=theta_fake,
            real_peak_sum=real_peak_sum,
            entropy_fake=entropy_fake,
            objective=objective,
            delta_xy=(dx, dy)
        )
        return obs, reward, terminated, truncated, info

# -----------------------------
#   简单测试 Env 是否工作
# -----------------------------

env = FakeLocationEnv(
    G_sig=G_sig,
    real_traj_xy=real_traj_xy,
    real_Peff_t=real_Peff_t,
    theta_grid=theta_grid,
    M=M, d_total=d_total, sigma2=sigma2,
    real_aoas=real_aoas_t,
    jammer_location=jammer_location,
    fake_power=fake_power,
    lambda_weight_fake=0.1,
    entropy_halfwin=1,
    fake_init_xy=fake_location,     # ★ 新增
    max_delta_pos=0.01,
)
print("\n[Env sanity check with random actions]")
obs, _ = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"t={env.t:2d}, action={info['fake_xy']}, reward={reward:.4f}, "
          f"real_peak_sum={info['real_peak_sum']:.4e}, H_fake={info['entropy_fake']:.4e}")
    if terminated:
        obs, _ = env.reset()

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# 创建带噪声的动作
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.05 * np.ones(n_actions)
)
# ==== Callback：每一个 time step 打印 reward ====
class PrintStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals 里有 "rewards", "infos" (vec_env)
        rewards = self.locals["rewards"]      # shape = (n_envs,)
        infos = self.locals["infos"]          # list 长度 n_envs

        for r, info in zip(rewards, infos):
            t = info.get("t", None)
            fake_aoa = info.get("fake_aoa", None)
            real_peak_sum = info.get("real_peak_sum", None)
            entropy_fake = info.get("entropy_fake", None)

            print(f"[train] global_step={self.num_timesteps:6d}, "
                  f"t={t}, reward={float(r): .4f}, "
                  f"fake_aoa={fake_aoa}, "
                  f"real_peak_sum={real_peak_sum:.4e}, "
                  f"H_fake={entropy_fake:.4e}")
        return True   # 返回 False 会中断训练
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    learning_rate=1e-3,
    batch_size=64,
    gamma=0.99,
    tau=0.001,

)

callback = PrintStepCallback()

model.learn(total_timesteps=10000)
fake_traj_all = np.array(env.fake_traj_current, dtype=float)
np.save("fake_traj_all.npy", fake_traj_all)
np.save("train_rewards.npy", np.array(env.reward))  # 保存训练过程中每一步的 reward

