from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Montando a rede bayesiana. Aqui vai basicamente o encadeamento das causas
# que podem acabar criando sintomas nos logs. Nada muito elaborado aqui.
model = DiscreteBayesianNetwork(
    [
        ("FalhaHardware", "FalhaNo"),
        ("FalhaHardware", "AlertaTemp"),
        ("ErroConfig", "FalhaNo"),
        ("FalhaNo", "LogKernelPanic"),
        ("FalhaNo", "Lentidao"),
        ("SobrecargaRede", "Lentidao"),
        ("Lentidao", "LogTimeout"),
    ]
)

# Probabilidades iniciais. Esses números são meio arbitrários,
# só pra dar algum comportamento razoável pro modelo.
cpd_hw = TabularCPD("FalhaHardware", 2, [[0.999], [0.001]])
cpd_conf = TabularCPD("ErroConfig", 2, [[0.99], [0.01]])
cpd_rede = TabularCPD("SobrecargaRede", 2, [[0.95], [0.05]])

# Temperatura costuma subir quando tem problema físico,
# mas tem casos que não esquentam, então ficou assim mesmo.
cpd_temp = TabularCPD(
    "AlertaTemp",
    2,
    [[0.99, 0.20], [0.01, 0.80]],
    evidence=["FalhaHardware"],
    evidence_card=[2],
)

# Aqui é onde o nó resolve morrer. Misturei hardware/config pra
# representar situações típicas de ambiente de produção quebrado.
cpd_no = TabularCPD(
    "FalhaNo",
    2,
    [[0.999, 0.10, 0.05, 0.01], [0.001, 0.90, 0.95, 0.99]],
    evidence=["FalhaHardware", "ErroConfig"],
    evidence_card=[2, 2],
)

# Kernel panic aparece só quando o nó já está ruim mesmo.
cpd_panic = TabularCPD(
    "LogKernelPanic",
    2,
    [[0.999, 0.40], [0.001, 0.60]],
    evidence=["FalhaNo"],
    evidence_card=[2],
)

# Aqui a lentidão depende da combinação "nó quebrado" + "rede atolada".
# Basicamente: se os dois estiverem ruins, já era.
cpd_slow = TabularCPD(
    "Lentidao",
    2,
    [[0.95, 0.10, 0.10, 0.01], [0.05, 0.90, 0.90, 0.99]],
    evidence=["FalhaNo", "SobrecargaRede"],
    evidence_card=[2, 2],
)

# Timeout: consequência direta de lentidão, então só depende disso.
cpd_timeout = TabularCPD(
    "LogTimeout",
    2,
    [[0.95, 0.20], [0.05, 0.80]],
    evidence=["Lentidao"],
    evidence_card=[2],
)

# Adiciona tudo na rede e verifica se não deixei nada quebrado.
model.add_cpds(
    cpd_hw, cpd_conf, cpd_rede, cpd_temp, cpd_no, cpd_panic, cpd_slow, cpd_timeout
)

# Se isso falhar, tem algo inconsistente nas tabelas.
assert model.check_model()

# Vou usar eliminação de variáveis pra inferência aqui.
infer = VariableElimination(model)

# ------------------------------------------------------------------
# CENÁRIO 1
# Timeout no log → tentando descobrir se a culpa é da rede ou hardware.
print("--- Cenário 1 ---")
resultado_rede = infer.query(variables=["SobrecargaRede"], evidence={"LogTimeout": 1})
resultado_hw = infer.query(variables=["FalhaHardware"], evidence={"LogTimeout": 1})
print(f"Probabilidade de ser Rede: {resultado_rede.values[1]:.4f}")
print(f"Probabilidade de ser Hardware: {resultado_hw.values[1]:.4f}")

# ------------------------------------------------------------------
# CENÁRIO 2
# Caso mais chato: panic + temperatura alta ao mesmo tempo.
print("\n--- Cenário 2 ---")
resultado = infer.query(
    variables=["FalhaHardware", "ErroConfig"],
    evidence={"LogKernelPanic": 1, "AlertaTemp": 1},
)
print(resultado)

# ------------------------------------------------------------------
# CENÁRIO 3
# MAP: dado panic + timeout, ele tenta apontar qual causa é mais provável.
print("\n--- Cenário 3 ---")
map_result = infer.map_query(
    variables=["FalhaHardware", "ErroConfig", "SobrecargaRede"],
    evidence={"LogTimeout": 1, "LogKernelPanic": 1},
)

# Ordenei só pra manter previsível a chave impressa.
ordenado = dict(sorted(map_result.items()))
resultado = next(iter(ordenado))
print("Causa mais provável:", resultado)
