import numpy as np
import matplotlib.pyplot as plt
import random

# --- System Parameters ---

M = 4  # Number of antennas at the array (Uniform Linear Array, ULA)
d_r = 2  # Number of real sources (true sensor nodes)
d_f = 1  # Number of fake sources (decoys we will optimize)
real_aoas = [20, 60]  # True AoAs (in degrees) of the real sources
sigma2 = 1e-3  # Noise power (variance of additive white Gaussian noise)
theta_grid = np.linspace(-90, 90, 181)  # Grid of scanning angles (-90° to 90°, 0.1° resolution)
d_over_lambda = 0.5  # Antenna spacing normalized to wavelength (half-wavelength spacing)
d_total = d_r + d_f  # Total number of sources MUSIC will detect (important!)

# generate location for jammer (global coords)
jammer_location = (0.0, 0.0)   # 阵列相位中心（GLOBAL）

# generate location for real and fake nodes (GLOBAL coords)
K = d_r + d_f
Lroom = 10.0   # 圆心 x
Wroom = 20.0   # 圆心 y
radius = 10.0       # 圆半径 (m)

nodes_xy = np.zeros((K, 2), dtype=float)
center = np.array([Lroom, Wroom], dtype=float)

rng = np.random.default_rng()  # 可加种子 rng = np.random.default_rng(123)

for k0 in range(K):
    # 正确的“圆盘内均匀采样”：r = R * sqrt(U), angle ~ U[0, 2π]
    r = radius * np.sqrt(rng.uniform(0.0, 1.0))
    angle = rng.uniform(0.0, 2*np.pi)
    offset = np.array([r*np.cos(angle), r*np.sin(angle)], dtype=float)
    pt = center + offset  # GLOBAL coords
    nodes_xy[k0, :] = pt


print("Node Locations (GLOBAL, x,y):\n", nodes_xy)
fig, ax = plt.subplots(figsize=(7, 7))

# 参考坐标轴
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)

# ULA（jammer 所在）位置
ax.scatter([jammer_location[0]], [jammer_location[1]],
           marker='s', s=100, label='ULA (jammer center)')

# real / fake 节点（按你 nodes_xy 的拆分方式：前 d_r 是 real，后 d_f 是 fake）
real_location = nodes_xy[:d_r, :]
fake_location = nodes_xy[d_r:, :]

ax.scatter(real_location[:, 0], real_location[:, 1],
           marker='o', s=60, label=f'Real (n={d_r})')
ax.scatter(fake_location[:, 0], fake_location[:, 1],
           marker='x', s=80, label=f'Fake (n={d_f})')

# 随机圆区域边界（圆心 (Lroom, Wroom)，半径 R）
theta = np.linspace(0, 2*np.pi, 256)
circle_x = Lroom + radius * np.cos(theta)
circle_y = Wroom + radius * np.sin(theta)
ax.plot(circle_x, circle_y, linewidth=1.0, label='Sampling circle')
ax.scatter([Lroom], [Wroom], marker='+', s=80, label='Circle center')

# 外观
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Global frame: ULA (jammer), real/fake nodes and sampling circle')
ax.grid(True)
ax.legend(loc='best')
plt.tight_layout()
#plt.show()

# 1) 定义：全局 -> 阵列坐标（阵列=jammer 在 jammer_location）
ARRAY_CENTER_GLOBAL = np.array(jammer_location, dtype=float)
ARRAY_YAW_DEG = 0.0  # 如果 ULA 不是沿全局 x 轴，填入偏航角（度）

def to_array_frame(points_xy, array_center=ARRAY_CENTER_GLOBAL, yaw_deg=ARRAY_YAW_DEG):
    pts = np.asarray(points_xy, dtype=float)                  # Nx2
    pts_shift = pts - np.asarray(array_center)[None, :]       # 平移：阵列中心作为原点
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(-yaw), np.sin(-yaw)                         # 旋转：把 ULA 对齐到 x 轴
    Rm = np.array([[c, -s], [s,  c]])
    return (pts_shift @ Rm.T)                                  # 返回阵列坐标系下的 (x', y')

# 2) 把 real 的位置从“全局”变到“阵列坐标系”
real_xy_array = to_array_frame(real_location)  # shape: (d_r, 2)

# 3) 计算 AoA：θ = atan2(x', y') （注意是 atan2( x', y' )，broadside=+y）
#real_aoas = list(np.degrees(np.arctan2(real_xy_array[:, 0], real_xy_array[:, 1])))
"注意，这里也可以直接来设置，比如 real_aoas = [20, 60]"
# 4) 把 fake 的位置也变到“阵列坐标系”，并计算 real/fake 到阵列的距离
fake_xy_array = to_array_frame(fake_location)  # shape: (d_f, 2)

# print("Real locations (ARRAY frame):\n", real_xy_array)
# print("Real AoAs (deg) w.r.t. jammer ULA:", real_aoas)
jammer_x, jammer_y = jammer_location

# real 的直线距离（到 jammer/ULA 相位中心）
real_dist = np.hypot(real_location[:, 0] - jammer_x,
                     real_location[:, 1] - jammer_y)      # shape: (d_r,)

# fake 的直线距离（你 d_f = 1）
fake_dist = np.hypot(fake_location[:, 0] - jammer_x,
                         fake_location[:, 1] - jammer_y)   # shape: (d_f,)
print("Real distances to ULA (m):", real_dist)
print("Fake distances to ULA (m):", fake_dist)
print("Real aoa",real_aoas)

real_power = 1.0  # Power of each real source
fake_power = 1.0  # Power of each fake source

# --- Helper Functions ---

def steering_vector(theta, M, d_over_lambda=0.5):
    """Compute the array steering vector for a given angle theta."""
    m = np.arange(M)  # Antenna element indices: 0,1,...,M-1
    return np.exp(-1j * 2 * np.pi * d_over_lambda * m * np.sin(np.radians(theta)))

def compute_covariance_matrix(real_aoas, fake_aoas, M, sigma2, real_power=1.0, fake_power=1.0):
    """
    Build the array covariance matrix:
    R = sum over real sources + sum over fake sources + noise.
    Each source contributes a rank-1 matrix scaled by its power.
    """
    all_aoas = real_aoas + list(fake_aoas)
    #all_powers = [real_power] + [fake_power]
    all_powers = np.concatenate([real_power, fake_power])
    all_powers = all_powers.tolist()
    # print(real_aoas)
    # print(fake_aoas)
    # print(real_power)
    # print(fake_power)
    # print(all_aoas)
    # print(all_powers)
    R = sigma2 * np.eye(M, dtype=complex)  # Start with noise
    for theta, power in zip(all_aoas, all_powers):
        a = steering_vector(theta, M)
        R += power * np.outer(a, np.conj(a))  # Add rank-1 contribution for each source

    # Add perturbation error
    Error = generate_perturbation_matrix(M)  # Adjust scale if needed
    R = R + Error

    return R

def generate_perturbation_matrix(M):
    """Generate a small Hermitian complex matrix E to perturb R."""
    A = (np.random.randn(M, M) + 1j * np.random.randn(M, M)) / np.sqrt(2)
    scale = 0 #1e-2
    E = scale * (A + A.conj().T) / 2  # Make it Hermitian
    return E

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

# --- Random Search for Optimal Fake AoAs ---

fake_grid = np.linspace(-90, 90, 181)  # Fake AoA candidates, spaced by 1 degree
best_entropy = -np.inf  # Initialize best entropy value
best_objective = np.inf
best_fake_aoas = None  # Initialize best fake AoA set
num_trials = 5000  # Number of random sets to try (controls search thoroughness)

#print("Starting random search for 8 fake AoAs...")
print(f"Starting random search for {d_f} fake AoAs...")

# Add perturbation error
#Error = generate_perturbation_matrix(M)  # Adjust scale if needed
#rror_abs = np.abs(Error)
#print(f"Error covariance matrix:", Error_abs )

for _ in range(num_trials):
    fake_aoas = random.sample(list(fake_grid), d_f)  # Randomly select 8 fake AoAs

 #   R = compute_covariance_matrix(real_aoas, fake_aoas, M, sigma2, real_power, fake_power)
 #   R = R + Error
 #   music_spectrum = compute_music_spectrum(R, M, d_total, theta_grid)
 #   unnormalized_music = music_spectrum.copy()  # Save before normalization
 #   music_spectrum /= np.sum(music_spectrum)  # Normalize the MUSIC spectrum to sum to 1
 #   entropy = compute_entropy(music_spectrum)  # Compute entropy of the normalized spectrum

 #   if entropy > best_entropy:
 #       best_entropy = entropy.copy()
 #       best_fake_aoas = fake_aoas.copy()  # Save the best fake AoAs that maximize entropy
 #       music_spectrum_final = music_spectrum.copy()
 #       unnormalized_music_final = unnormalized_music.copy()
    real_Peff = (real_power * np.ones_like(real_dist)) / np.maximum(real_dist, 1e-12)    # shape = (d_r,)
    fake_Peff = (fake_power * np.ones_like(fake_dist)) / np.maximum(fake_dist, 1e-12)    # shape = (d_f,)
    # print(real_Peff)
    # print(fake_Peff)
    # Compute covariance matrix with current fake AoAs
    R = compute_covariance_matrix(real_aoas, fake_aoas, M, sigma2, real_Peff, fake_Peff)
    #R = compute_covariance_matrix(real_aoas, fake_aoas, M, sigma2, real_power, fake_power)

    # Compute MUSIC spectrum over full grid
    music_spectrum = compute_music_spectrum(R, M, d_total, theta_grid)

    unnormalized_music = music_spectrum.copy()  # Save before normalization
    music_spectrum /= np.sum(music_spectrum)  # Normalize the MUSIC spectrum to sum to 1
    entropy = compute_entropy(music_spectrum)  # Compute entropy of the normalized spectrum

    # --- 1. Sum of spectrum values at real AoAs (minimize this) ---
    real_peak_sum = 0
    for theta in real_aoas:
        idx = np.argmin(np.abs(theta_grid - theta))
        real_peak_sum += music_spectrum[idx]
     #   real_peak_sum /= np.sum(real_peak_sum)

    # --- 2. Entropy of spectrum values at fake AoAs (maximize this) ---
    fake_values = []
    for theta in fake_aoas:
        idx = np.argmin(np.abs(theta_grid - theta))
        fake_values.append(music_spectrum[idx])
    fake_values = np.array(fake_values)
    #print(fake_values)
    fake_values /= np.sum(fake_values)  # Normalize for entropy
    entropy_fake = compute_entropy(fake_values)

    # --- 3. Combined objective ---
    lambda_weight_fake = 0.1  # Tune this value as needed
    objective = real_peak_sum - lambda_weight_fake  * entropy_fake

    if objective < best_objective:
        best_objective = objective.copy()
        best_fake_aoas = fake_aoas.copy()
        best_entropy = entropy_fake.copy()
        music_spectrum_final = music_spectrum.copy()
        unnormalized_music_final = unnormalized_music.copy()
        best_real_peak_sum = real_peak_sum.copy()


    # --- Output Results ---
# Temporarily set NumPy to print full arrays
#np.set_printoptions(threshold=np.inf)

print("\n=== Random Search Result ===")
print("Optimal Fake AoAs (degrees):", best_fake_aoas)
print(f"Maximum Spectrum Entropy (bits): {best_entropy:.4f}")
print("Real Normalized MUSIC Spectrum:")
print(best_real_peak_sum)
print("Objective:")
print(best_objective)
#print("Unnormalized MUSIC spectrum (full array):")
#print(unnormalized_music_final)

# Get indices of the top d_total largest values in descending order
top_indices = np.argsort(unnormalized_music_final)[-d_total:][::-1]

# Print the top values and corresponding angles
print(f"\nTop {d_total} values in unnormalized MUSIC spectrum:")
for idx in top_indices:
    theta = theta_grid[idx]
    value = unnormalized_music_final[idx]
    print(f"Theta = {theta:.2f}°, P(θ) = {value:.6e}")

#R_abs = np.abs(R)
#print(f"Covariance matrix:", R_abs )

# --- Compute Final MUSIC Spectrum with Best Fake Sources ---

#R_final = compute_covariance_matrix(real_aoas, best_fake_aoas, M, sigma2, real_power, fake_power)
#music_spectrum_final = compute_music_spectrum(R_final, M, d_total, theta_grid)
#music_spectrum_final /= np.sum(music_spectrum_final)  # Normalize again

# --- Print Spectrum Values at Real and Fake AoAs ---

print("\n=== Spectrum Values at Real and Fake AoAs ===")
all_sources = real_aoas + best_fake_aoas

for idx, src_theta in enumerate(all_sources):
    # Find the closest angle in theta_grid
    closest_idx = np.argmin(np.abs(theta_grid - src_theta))
    spectrum_value = music_spectrum_final[closest_idx]

    if idx < d_r:
        label = "Real Source"
    else:
        label = "Fake Source"

    print(f"{label}: theta = {src_theta:.2f} deg -> Spectrum = {spectrum_value:.6f}")

# --- Plot MUSIC Spectrum ---

plt.figure(figsize=(10, 6))
plt.plot(theta_grid, music_spectrum_final, label='MUSIC Spectrum', linewidth=2)
plt.scatter(real_aoas, [music_spectrum_final[np.argmin(np.abs(theta_grid - t))] for t in real_aoas],
            color='red', label='Real AoAs', marker='o', s=100)
plt.scatter(best_fake_aoas, [music_spectrum_final[np.argmin(np.abs(theta_grid - t))] for t in best_fake_aoas],
            color='green', label='Fake AoAs', marker='x', s=100)
# plt.title(f'MUSIC Spectrum with Optimized {d_f} Fake AoAs (Random Search)')
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Normalized Spectrum Value')
# plt.grid(True)
# plt.legend()
# plt.show()

# Set font sizes for labels and title
plt.title('MUSIC Spectrum with Optimized 8 Fake AoAs (Random Grid Search)', fontsize=16)
plt.xlabel('Angle (degrees)', fontsize=14)
plt.ylabel('Normalized Spectrum Value', fontsize=14)

# Adjust tick font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set legend font size
plt.legend(fontsize=12)

plt.grid(True)


# Legend on the right
#plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)

# Legend in top-right corner inside the plot
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()

# Save figure as PDF
plt.savefig('music_spectrum.pdf', format='pdf', bbox_inches='tight')

plt.show()

###########################

# --- Compute MUSIC Spectrum with Random Fake AoAs (for comparison) ---

# Randomly select 8 distinct fake AoAs from the fake grid
random_fake_aoas = random.sample(list(fake_grid), d_f)

# Build covariance matrix using the random fake AoAs
R_random = compute_covariance_matrix(real_aoas, random_fake_aoas, M, sigma2, real_Peff, fake_Peff)

# Compute MUSIC spectrum and normalize
music_spectrum_random = compute_music_spectrum(R_random, M, d_total, theta_grid)
unnormalized_music_random = music_spectrum_random.copy()  # Save before normalization
music_spectrum_random /= np.sum(music_spectrum_random)

# Compute and print entropy of random configuration
entropy_random = compute_entropy(music_spectrum_random)
print(f"\nEntropy of Random Fake AoAs Spectrum: {entropy_random:.4f} bits")

# --- Plot Comparison: Optimized vs Random Fake AoAs ---

plt.figure(figsize=(12, 6))

# Plot optimized spectrum
plt.plot(theta_grid, music_spectrum_final, label='Optimized Fake AoAs', linewidth=2)

# Plot random spectrum
plt.plot(theta_grid, music_spectrum_random, '--', label='Random Fake AoAs Spectrum', linewidth=2)

# Markers for real sources
plt.scatter(real_aoas, [music_spectrum_final[np.argmin(np.abs(theta_grid - t))] for t in real_aoas],
            color='red', label='Real AoAs', marker='o', s=80)

# Markers for optimized fake sources
plt.scatter(best_fake_aoas, [music_spectrum_final[np.argmin(np.abs(theta_grid - t))] for t in best_fake_aoas],
            color='green', label='Optimized Fake AoAs', marker='x', s=60)

# Markers for random fake sources
plt.scatter(random_fake_aoas, [music_spectrum_random[np.argmin(np.abs(theta_grid - t))] for t in random_fake_aoas],
            color='orange', label='Random Fake AoAs', marker='^', s=60)

plt.title('MUSIC Spectrum: Optimized vs Random Fake AoAs')
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalized Spectrum')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot Unnormalized MUSIC Spectrum ---

plt.figure(figsize=(12, 6))
plt.plot(theta_grid, unnormalized_music_final, label='Optimized Fake AoAs (Unnormalized)', linewidth=2)
plt.plot(theta_grid, unnormalized_music_random, '--', label='Random Fake AoAs (Unnormalized)', linewidth=2)

plt.title('Unnormalized MUSIC Spectrum: Optimized vs Random Fake AoAs')
plt.xlabel('Angle (degrees)')
plt.ylabel('Pseudo-Spectrum Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# === ADD-ON: Gym Environment  ===
# ================================
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    print("Gymnasium not installed. Install with: pip install gymnasium")
    raise

def _angles_to_obs(angles_deg):
    """把 real_aoas 编成 [cos, sin, cos, sin, ...] 作为状态（最小可用）。"""
    cs = []
    for th in angles_deg:
        r = np.deg2rad(th)
        cs.extend([np.cos(r), np.sin(r)])
    return np.asarray(cs, dtype=np.float32)

def _window_entropy_at_theta(music_norm, theta_grid, theta, half_win=1):
    """
    """
    idx = int(np.argmin(np.abs(theta_grid - theta)))
    lo = max(0, idx - half_win)
    hi = min(len(theta_grid) - 1, idx + half_win)
    p = music_norm[lo:hi+1].astype(float)
    p = p / max(p.sum(), 1e-12)
    p = np.clip(p, 1e-12, None)
    return -np.sum(p * np.log2(p))

class FakeAoAEnv(gym.Env):
    """
    单步环境：动作是在 theta_grid 上选一个假 AoA（离散动作）。
    奖励 = -(real_peak_sum - lambda_weight_fake * entropy_fake_window)
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 theta_grid,
                 M, d_total, sigma2,
                 real_aoas,          # list[deg]
                 real_Peff,          # shape=(d_r,)
                 fake_Peff,          # shape=(d_f,)  — 你目前 d_f=1
                 d_over_lambda=0.5,
                 lambda_weight_fake=0.1,
                 entropy_halfwin=1):
        super().__init__()
        self.theta_grid = np.asarray(theta_grid)
        self.M = M
        self.d_total = d_total
        self.sigma2 = sigma2
        self.real_aoas = list(real_aoas)
        self.real_Peff = np.asarray(real_Peff)      # 已含 pathloss 的等效功率
        self.fake_Peff = np.asarray(fake_Peff)      # 已含 pathloss 的等效功率
        self.d_over_lambda = d_over_lambda
        self.lambda_weight_fake = float(lambda_weight_fake)
        self.entropy_halfwin = int(entropy_halfwin)

        # 动作 = 在网格上选索引
        self.action_space = spaces.Discrete(len(self.theta_grid))
        # 状态 = [cos, sin]*d_r
        obs_dim = 2 * len(self.real_aoas)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = _angles_to_obs(self.real_aoas)
        return obs, {}

    def step(self, action):
       
        fake_theta = float(self.theta_grid[int(action)])
        fake_aoas = [fake_theta]

        # 用 “已含 pathloss 的功率” 进入你原 covariance 计算
        R = compute_covariance_matrix(self.real_aoas, fake_aoas,
                                      self.M, self.sigma2,
                                      real_power=self.real_Peff,
                                      fake_power=self.fake_Peff)
        music = compute_music_spectrum(R, self.M, self.d_total, self.theta_grid)
        music_norm = music / np.sum(music)

        # real_peaks 累加
        real_peak_sum = 0.0
        for th in self.real_aoas:
            idx = int(np.argmin(np.abs(self.theta_grid - th)))
            real_peak_sum += float(music_norm[idx])

        
        entropy_fake = _window_entropy_at_theta(music_norm, self.theta_grid,
                                                fake_theta, half_win=self.entropy_halfwin)

        objective = real_peak_sum - self.lambda_weight_fake * entropy_fake
        reward = -objective  

        obs = _angles_to_obs(self.real_aoas)
        info = dict(fake_aoa=fake_theta,
                    real_peak_sum=real_peak_sum,
                    entropy_fake=entropy_fake,
                    objective=objective)
        terminated = True   # 单步任务：选一次角度就结束
        truncated  = False
        return obs, reward, terminated, truncated, info

# 1) 复用你已计算好的 real_Peff / fake_Peff（含 pathloss），以及 real_aoas 等

real_Peff_env = (1.0 * np.ones_like(real_dist)) / np.maximum(real_dist, 1e-12)  # 等同你循环内的写法
fake_Peff_env = (1.0 * np.ones_like(fake_dist)) / np.maximum(fake_dist, 1e-12)

env = FakeAoAEnv(theta_grid=theta_grid,
                 M=M, d_total=d_total, sigma2=sigma2,
                 real_aoas=real_aoas,
                 real_Peff=real_Peff_env,
                 fake_Peff=fake_Peff_env,
                 d_over_lambda=d_over_lambda,
                 lambda_weight_fake=0.1,
                 entropy_halfwin=1)

# 随机 agent demo：采 10 次动作看看 reward（你可以换成 SB3 的 PPO/DDPG/TD3）
for _ in range(10):
    obs, info = env.reset()
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    print(f"[env] action theta={info['fake_aoa']:.1f}°, reward={reward:.4f}, "
          f"real_peak_sum={info['real_peak_sum']:.4e}, H_fake={info['entropy_fake']:.4e}")
