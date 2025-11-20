import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# HMM GAUSSIANO - FROM SCRATCH
# ============================


def log_multivariate_normal_density(x, mean, cov):
    """
    Calcula log p(x | mean, cov) para vetor x (1D) com cov (dxd).
    Implementado em log para estabilidade numérica.
    """
    d = x.shape[0]
    # regularização pequena para evitar cov singular
    eps = 1e-6
    cov_reg = cov + np.eye(d) * eps
    sign, logdet = np.linalg.slogdet(cov_reg)
    inv = np.linalg.inv(cov_reg)
    diff = x - mean
    return -0.5 * (d * np.log(2 * np.pi) + logdet + diff.T @ inv @ diff)


def initialize_parameters(X, n_components, random_state=42):
    """
    Inicialização simples: escolhe n_components pontos aleatórios como médias,
    covariância global replicada, A uniforme, pi uniforme.
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    pi = np.ones(n_components) / n_components
    A = np.ones((n_components, n_components))
    A /= A.sum(axis=1, keepdims=True)

    idx = np.random.choice(n_samples, size=n_components, replace=False)
    means = X[idx].copy()
    global_cov = np.cov(X, rowvar=False)
    covs = np.array([global_cov.copy() for _ in range(n_components)])
    return pi, A, means, covs


def forward_backward_log(X, pi, A, means, covs):
    """
    Forward-backward em log-space.
    Retorna gamma (posteriors) e xi (pairwise posteriors).
    """
    n_samples, n_features = X.shape
    K = len(pi)

    # log emissões: B[t, k] = log p(x_t | state=k)
    B = np.zeros((n_samples, K))
    for t in range(n_samples):
        for k in range(K):
            B[t, k] = log_multivariate_normal_density(X[t], means[k], covs[k])

    # log pi e log A
    log_pi = np.log(pi + 1e-16)
    log_A = np.log(A + 1e-16)

    # Forward (alpha) em log
    log_alpha = np.zeros((n_samples, K))
    log_alpha[0] = log_pi + B[0]
    for t in range(1, n_samples):
        # logsumexp sobre j: log(sum_j exp(log_alpha[t-1, j] + log_A[j, k]))
        for k in range(K):
            prev = log_alpha[t - 1] + log_A[:, k]
            # log-sum-exp
            m = np.max(prev)
            log_alpha[t, k] = B[t, k] + (m + np.log(np.sum(np.exp(prev - m))))

    # Backward (beta) em log
    log_beta = np.zeros((n_samples, K))
    log_beta[-1] = 0.0
    for t in range(n_samples - 2, -1, -1):
        for i in range(K):
            tmp = log_A[i, :] + B[t + 1, :] + log_beta[t + 1, :]
            m = np.max(tmp)
            log_beta[t, i] = m + np.log(np.sum(np.exp(tmp - m)))

    # Gamma: posterior de cada estado em cada tempo (em log -> exp)
    log_gamma = log_alpha + log_beta
    # normalizar
    log_gamma -= np.max(log_gamma, axis=1, keepdims=True) + np.log(
        np.sum(
            np.exp(log_gamma - np.max(log_gamma, axis=1, keepdims=True)),
            axis=1,
            keepdims=True,
        )
    )
    gamma = np.exp(log_gamma)

    # Xi: posterior de pares (t, i -> t+1, j)
    xi = np.zeros((n_samples - 1, K, K))
    for t in range(n_samples - 1):
        # log xi_{t,i,j} log_alpha[t,i] + log_A[i,j] + B[t+1,j] + log_beta[t+1,j]
        log_xi_t = (
            log_alpha[t][:, None] + log_A + B[t + 1][None, :] + log_beta[t + 1][None, :]
        )
        # normaliza
        m = np.max(log_xi_t)
        log_xi_t -= m + np.log(np.sum(np.exp(log_xi_t - m)))
        xi[t] = np.exp(log_xi_t)

    return gamma, xi, B


def baum_welch(X, n_components, n_iter=20, tol=1e-4, random_state=42):
    """
    Treina HMM Gaussiano com Baum-Welch (EM) em log-space.
    Retorna parâmetros treinados.
    """
    n_samples, n_features = X.shape
    pi, A, means, covs = initialize_parameters(
        X, n_components, random_state=random_state
    )
    prev_loglik = None

    for it in range(n_iter):
        gamma, xi, B = forward_backward_log(X, pi, A, means, covs)

        # M-step
        # pi <- gamma[0]
        pi = gamma[0].copy()
        # A <- sum_t xi_t / sum rows
        sum_xi = np.sum(xi, axis=0)  # KxK
        A = sum_xi / (np.sum(sum_xi, axis=1, keepdims=True) + 1e-16)

        for k in range(n_components):
            weight = gamma[:, k].sum() + 1e-16
            means[k] = (gamma[:, k][:, None] * X).sum(axis=0) / weight

            diff = X - means[k]
            # cov = sum_t gamma[t,k] * diff_t outer diff_t / weight
            cov_k = np.zeros((n_features, n_features))
            for t in range(n_samples):
                cov_k += gamma[t, k] * np.outer(diff[t], diff[t])
            cov_k /= weight
            # regularize
            covs[k] = cov_k + np.eye(n_features) * 1e-6

        _, _, B = forward_backward_log(X, pi, A, means, covs)

        log_pi = np.log(pi + 1e-16)
        log_A = np.log(A + 1e-16)
        log_alpha = np.zeros((n_samples, n_components))
        log_alpha[0] = log_pi + B[0]
        for t in range(1, n_samples):
            for k in range(n_components):
                prev = log_alpha[t - 1] + log_A[:, k]
                m = np.max(prev)
                log_alpha[t, k] = B[t, k] + (m + np.log(np.sum(np.exp(prev - m))))

        m = np.max(log_alpha[-1])
        loglik = m + np.log(np.sum(np.exp(log_alpha[-1] - m)))

        print(f"Iter {it + 1}/{n_iter} - loglik: {loglik:.4f}")

        if prev_loglik is not None and abs(loglik - prev_loglik) < tol:
            print("Convergido (tol atingida).")
            break
        prev_loglik = loglik

    return pi, A, means, covs


def viterbi_log(X, pi, A, means, covs):
    """
    Viterbi em log-space para observações contínuas (Gaussiano).
    Retorna sequência de estados (0..K-1).
    """
    n_samples, n_features = X.shape
    K = len(pi)
    log_pi = np.log(pi + 1e-16)
    log_A = np.log(A + 1e-16)

    # calcula log emissões
    B = np.zeros((n_samples, K))
    for t in range(n_samples):
        for k in range(K):
            B[t, k] = log_multivariate_normal_density(X[t], means[k], covs[k])

    T1 = np.zeros((K, n_samples))
    T2 = np.zeros((K, n_samples), dtype=int)

    T1[:, 0] = log_pi + B[0]
    for t in range(1, n_samples):
        for j in range(K):
            prob = T1[:, t - 1] + log_A[:, j]
            T2[j, t] = np.argmax(prob)
            T1[j, t] = prob[T2[j, t]] + B[t, j]

    states = np.zeros(n_samples, dtype=int)
    states[-1] = np.argmax(T1[:, -1])
    for t in range(n_samples - 1, 0, -1):
        states[t - 1] = T2[states[t], t]
    return states


# ============================
# EXEMPLO: Usando dados financeiros
# ============================

if __name__ == "__main__":
    # Coleta de dados
    data = yf.download("^BVSP", start="2020-01-01", end="2024-01-01", progress=False)

    df = data[["Close"]].copy()
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["Return"].rolling(window=10).std()
    df.dropna(inplace=True)

    X = df[["Return", "Volatility"]].values

    # configuração do modelo
    n_components = 3
    n_iter = 20

    pi, A, means, covs = baum_welch(
        X, n_components=n_components, n_iter=n_iter, random_state=42
    )

    # predição de estados (Viterbi)
    states = viterbi_log(X, pi, A, means, covs)
    df["State"] = states

    print("\nMédias por estado (Return, Volatility):")
    for k in range(n_components):
        print(f"Estado {k}: mean={means[k]}, cov diag={np.diag(covs[k])}")

    plt.figure(figsize=(14, 8))
    colors = ["red", "green", "blue", "orange", "purple"]
    for k in range(n_components):
        subset = df[df["State"] == k]
        plt.scatter(
            subset.index, subset["Close"], s=10, c=colors[k], label=f"Estado {k}"
        )

    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.legend()
    plt.grid(True)
    plt.show()
