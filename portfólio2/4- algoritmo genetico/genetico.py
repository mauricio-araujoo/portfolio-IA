import numpy as np
import math


# cada usina tem um custo diferente e limites de geração (min, max).
PLANT_DATA = [
    {'a': 0.004, 'b': 5.0, 'c': 500, 'min': 50, 'max': 200}, # usina 1 
    {'a': 0.006, 'b': 7.0, 'c': 800, 'min': 30, 'max': 300}, # usina 2
    {'a': 0.009, 'b': 10.0, 'c': 1200, 'min': 20, 'max': 150} # usina 3
]

N_PLANTS = len(PLANT_DATA)
TOTAL_DEMAND = 500  # energia necessaria para a cidade
PENALTY_FACTOR = 100000.0  # penalidade por não atender a demanda acima

# limite
MIN_BOUNDS = np.array([p['min'] for p in PLANT_DATA])
MAX_BOUNDS = np.array([p['max'] for p in PLANT_DATA])

def calculate_cost_and_brightness(position):
    """
    calcula o brilho e partir do custo de cada vagalume
    """
    total_cost = 0
    total_generation = 0
    
    # calculo custo
    for i in range(N_PLANTS):
        P = position[i]
        plant = PLANT_DATA[i]
        cost = plant['a'] * P**2 + plant['b'] * P + plant['c']
        total_cost += cost
        total_generation += P
        
    # aplica penalidade
    demand_violation = abs(total_generation - TOTAL_DEMAND)
    penalty = PENALTY_FACTOR * demand_violation**2  # penalidade^2 pra atender mais rapido demanda
    
    final_cost = total_cost + penalty
    
    # inverter porque queremos menor custo
    brightness = 1.0 / (1.0 + final_cost)
    
    return final_cost, brightness


# parametros do Algoritmo
N_FIREFLIES = 50      # numero de candidatas
MAX_GEN = 300         # iteracoes
BETA_0 = 1.0          # atratividade
GAMMA = 0.5           # quão rapido a atracao diminui
ALPHA = 0.2           # movimento aleatorio

# cria vagalumes
fireflies_pos = np.zeros((N_FIREFLIES, N_PLANTS))
fireflies_cost = np.full(N_FIREFLIES, np.inf)
fireflies_brightness = np.zeros(N_FIREFLIES)

for i in range(N_FIREFLIES):
    # cria solução aleatoria valida
    fireflies_pos[i] = MIN_BOUNDS + np.random.rand(N_PLANTS) * (MAX_BOUNDS - MIN_BOUNDS)
    cost, brightness = calculate_cost_and_brightness(fireflies_pos[i])
    fireflies_cost[i] = cost
    fireflies_brightness[i] = brightness

# melhor solução encontrada ate agora
best_index = np.argmin(fireflies_cost)
best_cost = fireflies_cost[best_index]
best_position = fireflies_pos[best_index].copy()

print(f"Iniciando simulação... Demanda Alvo: {TOTAL_DEMAND} MW")
print(f"Melhor custo inicial (aleatório): ${best_cost:.2f}")

# geracao ate agr
for gen in range(MAX_GEN):
    
    for i in range(N_FIREFLIES):
        for j in range(N_FIREFLIES):
            
            if fireflies_brightness[j] > fireflies_brightness[i]:
                
                # distância euclidiana ao quadrado
                r_squared = np.sum((fireflies_pos[i] - fireflies_pos[j])**2)
                
                # beta diminui pela atratividade com a absorsao da luz
                beta = BETA_0 * math.exp(-GAMMA * r_squared)
                
                # passo aleatorio
                random_step = ALPHA * (np.random.rand(N_PLANTS) - 0.5) * (MAX_BOUNDS - MIN_BOUNDS)
                
                # move o menos brilhoso pro mais
                fireflies_pos[i] += beta * (fireflies_pos[j] - fireflies_pos[i]) + random_step
                
                #limites
                fireflies_pos[i] = np.clip(fireflies_pos[i], MIN_BOUNDS, MAX_BOUNDS)
                
                # novo custo e brilho
                cost, brightness = calculate_cost_and_brightness(fireflies_pos[i])
                fireflies_cost[i] = cost
                fireflies_brightness[i] = brightness
                
                # atualiza melhor global
                if cost < best_cost:
                    best_cost = cost
                    best_position = fireflies_pos[i].copy()

    if (gen + 1) % 50 == 0:
        print(f"Geração {gen + 1}/{MAX_GEN}, Melhor Custo: ${best_cost:.2f}")

print("\n--- Simulação Concluída ---")
print(f"Melhor Custo encontrado: ${best_cost:.2f}")
print("Distribuição de Energia Ótima:")

total_gen_final = 0
for i in range(N_PLANTS):
    gen_power = best_position[i]
    total_gen_final += gen_power
    plant = PLANT_DATA[i]
    print(f"  Usina {i+1}: {gen_power:.2f} MW (Limites: [{plant['min']}, {plant['max']}])")

print(f"\nGeração Total: {total_gen_final:.2f} MW")
print(f"Demanda Alvo:   {TOTAL_DEMAND:.2f} MW")
print(f"Diferença (Erro): {abs(total_gen_final - TOTAL_DEMAND):.4f} MW")
