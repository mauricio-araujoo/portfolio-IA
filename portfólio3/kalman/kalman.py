import matplotlib.pyplot as plt
import numpy as np


def filtro_kalman_financeiro(data, R_noise, Q_noise):
    """
    Aplica um Filtro de Kalman 1D em uma série temporal financeira.
    """

    # --- 1. Inicialização ---
    n_iter = len(data)
    sz = (n_iter,)

    # Alocar espaço para os arrays
    xhat = np.zeros(sz)
    P = np.zeros(sz)
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)

    # Chutes iniciais
    xhat[0] = data[0]
    P[0] = 1.0

    # A = 1 (O valor de amanhã é baseado no de hoje)
    # H = 1 (Medimos o preço diretamente)

    # --- 2. Loop do Filtro ---
    for k in range(1, n_iter):
        # Projetamos o estado à frente
        xhatminus[k] = xhat[k - 1]
        # Projetamos a incerteza à frente
        Pminus[k] = P[k - 1] + Q_noise

        # 1. Calcular o Ganho de Kalman

        K[k] = Pminus[k] / (Pminus[k] + R_noise)

        # 2. Atualizar a estimativa com a medição atual (data[k])
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])

        # 3. Atualizar a incerteza do erro
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


# --- Simulação de Dados ---
np.random.seed(42)
t = np.linspace(0, 100, 100)

# Criar uma "Tendência Real" (ex: subida lenta com uma queda no meio)
valor_real = np.linspace(50, 80, 100)
valor_real[50:] -= np.linspace(0, 15, 50)
# Adicionar "Ruído de Mercado" (Volatilidade diária)
ruido = np.random.normal(0, 4, 100)
precos_mercado = valor_real + ruido

# --- Executar o Filtro ---
estimativa_kalman = filtro_kalman_financeiro(precos_mercado, R_noise=16.0, Q_noise=0.1)

# --- Visualização ---
plt.figure(figsize=(12, 6))
plt.plot(precos_mercado, "k.", alpha=0.5, label="Preço de Mercado (Observado/Ruído)")
plt.plot(valor_real, "g--", label="Valor Real (Tendência Econômica - Desconhecida)")
plt.plot(estimativa_kalman, "b-", linewidth=2, label="Filtro de Kalman (Estimado)")

plt.xlabel("Dias")
plt.ylabel("Preço ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
