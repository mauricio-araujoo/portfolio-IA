import math
import time

# jogadores
PLAYER_AI = 'X'
PLAYER_HUMAN = 'O'
EMPTY = ' '

def print_board(board):
    """printa o jogo"""
    print("\n  0 1 2")
    for i, row in enumerate(board):
        print(f"{i} {'|'.join(row)}")
        if i < 2:
            print("  -----")
    print()

def check_winner(board):
    """verifica vitoria ou empate"""
    
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    if all(board[i][j] != EMPTY for i in range(3) for j in range(3)):
        return 'DRAW'
    # volta none se jogo ainda rodando
    return None

def get_available_moves(board):
    """volta os movimentos possiveis para o algoritmo"""
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.append((i, j))
    return moves

def minimax(board, depth, is_maximizing, alpha, beta):
    """
    minimax com poda Alfa-Beta.
    'is_maximizing' é True para a IA ('X') e False para o Humano ('O').
    'alpha' é a melhor pontuação encontrada para o Maximizador.
    'beta' é a melhor pontuação encontrada para o Minimizador.
    """
    
    result = check_winner(board)
    if result:
        if result == PLAYER_AI:
            return 10 - depth  # peso para vitoria da ia
        elif result == PLAYER_HUMAN:
            return depth - 10 # peso da derrota
        elif result == 'DRAW':
            return 0

    if is_maximizing:
        best_score = -math.inf
        for (r, c) in get_available_moves(board):
            board[r][c] = PLAYER_AI  # joga
            
            score = minimax(board, depth + 1, False, alpha, beta)
            
            board[r][c] = EMPTY      # ve se jogada foi ruim então volta
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            
            if beta <= alpha:
                break  # descarta os resoltados ruins
                
        return best_score

    # poda para jogadas do jogador humano
    else:
        best_score = math.inf
        for (r, c) in get_available_moves(board):
            board[r][c] = PLAYER_HUMAN # joga
            
            score = minimax(board, depth + 1, True, alpha, beta)
            
            board[r][c] = EMPTY # bt
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            
            if beta <= alpha:
                break  #descarte 
                
        return best_score

def find_best_move(board):
    """
    função pro melhor movimento
    """
    best_score = -math.inf
    best_move = None
    
    for (r, c) in get_available_moves(board):
        board[r][c] = PLAYER_AI # joga
        
        move_score = minimax(board, 0, False, -math.inf, math.inf)
        
        board[r][c] = EMPTY     # volta jogada
        
        if move_score > best_score:
            best_score = move_score
            best_move = (r, c)
            
    return best_move

def play_game():
    """loop do jogo."""
    board = [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]
    
    current_player = PLAYER_AI # IA COMEÇA SÓ PRA CONSTAR QUE N CONSEGUI GANHAR NENHUMA VEZ 
    
    while True:
        print_board(board)
        
        if current_player == PLAYER_HUMAN:
            try:
                row = int(input("Digite a linha (0, 1, ou 2): "))
                col = int(input("Digite a coluna (0, 1, ou 2): "))
                
                if 0 <= row <= 2 and 0 <= col <= 2:
                    if board[row][col] == EMPTY:
                        board[row][col] = PLAYER_HUMAN
                        current_player = PLAYER_AI
                    else:
                        print("Posição já ocupada. Tente novamente.")
                else:
                    print("Posição inválida. Use números de 0 a 2.")
            except ValueError:
                print("Entrada inválida. Digite números.")
        
        else:
            print("IA ('X') está pensando...")
            
            move = find_best_move(board)
            if move:
                board[move[0]][move[1]] = PLAYER_AI
                print(f"IA jogou na posição {move}")
                current_player = PLAYER_HUMAN
        
        winner = check_winner(board)
        if winner:
            print_board(board)
            if winner == 'DRAW':
                print("O jogo terminou em EMPATE!")
            else:
                print(f"O jogador '{winner}' VENCEU!")
            break

if __name__ == "__main__":
    play_game()
