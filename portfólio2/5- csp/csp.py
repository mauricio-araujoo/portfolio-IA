import time

N = 9  # tamanho do lado do tabuleiro ^2
BOX_SIZE = 3 # tamanho dos blocos ^2

def imprimir_tabuleiro(tabuleiro):
    """printa tabuleiro"""
    print("+-------+-------+-------+")
    for i in range(N):
        if i > 0 and i % BOX_SIZE == 0:
            print("|-------+-------+-------|")
        
        linha_str = "| "
        for j in range(N):
            if j > 0 and j % BOX_SIZE == 0:
                linha_str += "| "
            
            val = tabuleiro[i][j]
            if val == 0:
                linha_str += ". "
            else:
                linha_str += f"{val} "
                
        print(linha_str + "|")
    print("+-------+-------+-------+")


def encontrar_vazio(tabuleiro):
    """procura celula vazia"""
    for i in range(N):
        for j in range(N):
            if tabuleiro[i][j] == 0:
                return (i, j)  # Retorna (linha, coluna)
    return None  # Retorna None se o tabuleiro estiver completo


def eh_seguro(tabuleiro, linha, col, num):
    """
    ve se numero a ser colocado n inflinge as regras
    """
    
    # verifica linha
    for j in range(N):
        if tabuleiro[linha][j] == num and j != col:
            return False
            
    # verifica coluna
    for i in range(N):
        if tabuleiro[i][col] == num and i != linha:
            return False
            
    # procura dentro do bloco 
    inicio_linha_bloco = (linha // BOX_SIZE) * BOX_SIZE
    inicio_col_bloco = (col // BOX_SIZE) * BOX_SIZE
    
    for i in range(BOX_SIZE):
        for j in range(BOX_SIZE):
            if tabuleiro[inicio_linha_bloco + i][inicio_col_bloco + j] == num and \
               (inicio_linha_bloco + i != linha or inicio_col_bloco + j != col):
                return False
                
    # true se seguro
    return True


def resolver_sudoku(tabuleiro):
    """
    funcao backtracking
    """
    
    pos_vazia = encontrar_vazio(tabuleiro)
    
    # se preenchido ta pronto
    if not pos_vazia:
        return True
        
    linha, col = pos_vazia
    
    # preenche de 1 a 9
    for num in range(1, N + 1):
        
        # valida regras
        if eh_seguro(tabuleiro, linha, col, num):
            
            # joga
            tabuleiro[linha][col] = num
            
            # vai pra proxima
            if resolver_sudoku(tabuleiro):
                return True  # propaga se numero colocado for bom
                
            tabuleiro[linha][col] = 0 # desfaz se deu errado
            
    # retorna falso pro bt voltar
    return False


tabuleiro_exemplo = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

print("--- Tabuleiro Original (9x9) ---")
imprimir_tabuleiro(tabuleiro_exemplo)

print("\n...Resolvendo...\n")
start_time = time.time()

if resolver_sudoku(tabuleiro_exemplo):
    end_time = time.time()
    print(f"--- Solução Encontrada (em {end_time - start_time:.4f} segundos) ---")
    imprimir_tabuleiro(tabuleiro_exemplo)
else:
    print("Não foi encontrada solução para este Sudoku.")
