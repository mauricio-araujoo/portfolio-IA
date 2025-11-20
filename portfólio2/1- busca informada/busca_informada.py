import math
import heapq


# Este dicionário simula um modelo de tradução.
dicionario = {
    '<s>': {
        'O': 0.40,
        'A': 0.35,
        'Ela': 0.15
    },
    'O': {
        'paciente': 0.30,
        'enfermeiro': 0.20,
        'médico': 0.1
    },
    'A': {
        'enfermeira': 0.50,
        'paciente': 0.10,
        'médica': 0.05
    },
    'paciente': {
        'falou': 0.40,
        'ouviu': 0.20,
        '</s>': 0.1
    },
    'enfermeiro': {
        'falou': 0.50,
        'viu': 0.30,
        '</s>': 0.1
    },
    'enfermeira': {
        'falou': 0.60,
        'disse': 0.30,
        '</s>': 0.05
    },
    'falou': {
        'com': 0.70,
        'para': 0.20,
        '</s>': 0.1
    },
    'com': {
        'o': 0.50,
        'a': 0.40,
        '</s>': 0.1
    },
    'o': {
        'paciente': 0.80,
        'enfermeiro': 0.1,
        '</s>': 0.1
    },
    'a': {
        'paciente': 0.30,
        'enfermeira': 0.6,
        '</s>': 0.1
    }
}

def beam_search(model, beam_width, max_len=10):
    """
    Implementa o Beam Search.
    """

    start_token = '<s>'
    start_sequence = [start_token]

    # O 'feixe' (beam) armazena as k melhores hipóteses atuais
    current_beam = [(0.0, start_sequence)]

    # Armazena hipóteses que já encontraram
    completed_sequences = []

    for _ in range(max_len):

        # Lista para todos os novos candidatos
        all_candidates = []

        # Expande cada hipótese no feixe atual
        for log_score, seq in current_beam:

            # Se a sequência já terminou vai para as 'completas'
            last_word = seq[-1]
            if last_word == '</s>':
                completed_sequences.append((log_score, seq))
                continue

            # Pega as próximas palavras possíveis do modelo
            next_word_probs = model.get(last_word, {})

            for word, prob in next_word_probs.items():
                # Calcula o novo score
                new_log_score = log_score + math.log(prob)
                new_seq = seq + [word]
                all_candidates.append((new_log_score, new_seq))

        # Se não houver novos candidatos, paramos
        if not all_candidates:
            break

        # Ordena todos os candidatos gerados e seleciona os 'k' melhores.
        current_beam = heapq.nlargest(beam_width, all_candidates, key=lambda x: x[0])

        # Se todas as hipóteses no feixe foram para 'completas'
        if not current_beam:
            break

    completed_sequences.extend(current_beam)

    # Normaliza os scores pelo comprimento para evitar que frases curtas sejam sempre preferidas
    final_results = []
    for log_score, seq in completed_sequences:
        if len(seq) > 1:
            normalized_score = log_score / (len(seq) - 1)
        else:
            normalized_score = -float('inf')

        final_results.append((normalized_score, log_score, seq))

    final_results.sort(key=lambda x: x[0], reverse=True)

    return final_results


def printar_resultados(result):
    """Função auxiliar para imprimir o resultado de forma legível."""
    if not result:
        return "Nenhum resultado encontrado."

    best_norm_score, best_log_score, best_seq = result[0]
    # Converte a lista de palavras em uma string, ignorando <s> e </s>
    sentence = " ".join(word for word in best_seq if word not in ['<s>', '</s>'])

    return f"Tradução: '{sentence}' "



if __name__ == "__main__":

    print("--- Problema da Tradução: 'The nurse spoke to the patient.' ---")

    print("\nExecutando Beam Search como busca gulosa (k=1):")
    beam_resultado = beam_search(dicionario, beam_width=1)
    print(printar_resultados(beam_resultado))

    print("\nExecutando Beam Search (k=3):")
    beam_resultado = beam_search(dicionario, beam_width=3)
    print(printar_resultados(beam_resultado))
