# gauss_markov_channel.py
import numpy as np

def gauss_markov_cx(N: int, M: int, ite: int, Lambda: float, seed: int | None = None) -> np.ndarray:
    """
    复高斯一阶 Gauss–Markov（AR(1)）信道:
        G[t] = sqrt(1 - Lambda**2) * W[t] + Lambda * G[t-1]
    """

    G_sig = np.empty((N, M, ite), dtype=np.complex64)

    # 初值：noise_init_G
    noise_init_G = np.random.normal(0, np.sqrt(0.5), (N, M)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                   (N, M))

    for t in range(ite):
        new_noise_G = np.random.normal(0, np.sqrt(0.5), (N, M)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                   (N, M))

        G = np.sqrt(1-Lambda**2) * new_noise_G + Lambda * noise_init_G
        noise_init_G = G
        G_sig[:, :, t] = G

    return G_sig

if __name__ == "__main__":
    N = 3
    M = 1
    ite = 20000
    Lambda = 0.9
    seed = 3407

    G_sig = gauss_markov_cx(N, M, ite, Lambda, seed)
    print(G_sig.shape)  # 输出 (N, M, ite)
    np.save("gauss_markov_channel_rho0.9.npy", G_sig)  # 保存为 .npy 文件

