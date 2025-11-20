import itertools

class Node:
    """
    Representa um nó do tabuleiro e seus estados
    """
    def __init__(self, state, parent=None, action=None, depth=0):
        # O estado é uma tupla de tuplas
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def find_blank(self):
        """encontra a posicao do espaço vazio"""
        for r in range(3):
            for c in range(3):
                if self.state[r][c] == 0:
                    return r, c
        return None # Não deve acontecer

    def get_path(self):
        """retorna o caminho do nó inicial ate esse"""
        path = []
        current = self
        while current.parent:
            path.append(current.action)
            current = current.parent
        return path[::-1] # inverte para ter a ordem do início ao fim

    def __str__(self):
        """printa o tabuleiro"""
        return '\n'.join([' '.join(map(str, row)) for row in self.state]) + '\n'

    def __eq__(self, other):
        """procura no repetido"""
        return self.state == other.state
    
    def __hash__(self):
        """transforma pra adicionar no set"""
        return hash(self.state)

def get_neighbors(node):
    """
    retorna os nós vizinhos ou seja possiveis caminhos
    """
    neighbors = []
    r, c = node.find_blank()
    
    # movimentos possíveis
    moves = [('Cima', r - 1, c), 
             ('Baixo', r + 1, c), 
             ('Esquerda', r, c - 1), 
             ('Direita', r, c + 1)]

    for action, nr, nc in moves:
        # verifica se o movimento nao sai do tabuleiro
        if 0 <= nr < 3 and 0 <= nc < 3:
            # cria a grade pra poder fazer o proximo movimento e guardar o estado atual
            new_grid = [list(row) for row in node.state]
            
            new_grid[r][c], new_grid[nr][nc] = new_grid[nr][nc], new_grid[r][c]
            
            new_state_tuple = tuple(tuple(row) for row in new_grid)
            
            neighbor_node = Node(new_state_tuple, 
                                 parent=node, 
                                 action=action, 
                                 depth=node.depth + 1)
            neighbors.append(neighbor_node)
            
    return neighbors

def dls(node, goal_state, limit, path_visited):
    """
    busca em profundidade limitada auxiliar para busca iterativa
    """
    if node.state == goal_state:
        return node
    
    if node.depth >= limit:
        return None

    path_visited.add(node.state)
    
    for neighbor in get_neighbors(node):
        if neighbor.state not in path_visited:
            result = dls(neighbor, goal_state, limit, path_visited)
            if result:
                return result

    path_visited.remove(node.state)
    return None

def iddfs(initial_state_tuple, goal_state_tuple):
    """
    Busca em Profundidade Iterativa (Iterative Deepening DFS).
    """
    initial_node = Node(initial_state_tuple)
    
    for limit in itertools.count():
        print(f"Tentando com profundidade limite: {limit}")
        
        path_visited = set()
        
        result = dls(initial_node, goal_state_tuple, limit, path_visited)
        
        if result:
            print(f"\nSolução encontrada na profundidade {limit}!")
            return result
        
        if limit > 31:
            print("Limite máximo de profundidade (31) atingido. Parando.")
            return None


GOAL_STATE = ((1, 2, 3),
              (4, 5, 6),
              (7, 8, 0))

INITIAL_STATE = ((1, 2, 3),
                 (4, 0, 5),
                 (7, 8, 6))

print("--- 8-Puzzle Solver ---")
print("\nEstado Inicial:")
print(Node(INITIAL_STATE))
print("Estado Alvo:")
print(Node(GOAL_STATE))

solution_node = iddfs(INITIAL_STATE, GOAL_STATE)

if solution_node:
    path = solution_node.get_path()
    print(f"Caminho da solução: {path}")
    print(f"Total de movimentos: {len(path)}\n")
    
    current = solution_node
    steps = []
    while current:
        steps.append(current)
        current = current.parent
    
    for i, step in enumerate(reversed(steps)):
        if i == 0:
            print("--- Início ---")
        else:
            print(f"--- Movimento {i} ({step.action}) ---")
        print(step)
        
else:
    print("Nenhuma solução encontrada (ou o puzzle é insolúvel).")
