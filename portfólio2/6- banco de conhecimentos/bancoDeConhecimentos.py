import collections

class KnowledgeBase:
    """
    simulacao banco de conhecimentos
    """
    def __init__(self):
        # armazena dados sobre individuos
        self.abox = collections.defaultdict(set)

        # armazena regras
        self.tbox = collections.defaultdict(set)

    def add_fact(self, individual: str, concept: str):
        """adicao de fatos no ABox."""
        print(f"[Fato Adicionado]  INDIVÍDUO: '{individual}' é um '{concept}'.")
        self.abox[individual].add(concept)

    def add_rule(self, sub_concept: str, super_concept: str):
        """
        adiciona regras no TBox
        """
        print(f"[Regra Adicionada] REGRA: Todo '{sub_concept}' é um '{super_concept}'.")
        self.tbox[sub_concept].add(super_concept)

    def get_all_concepts(self, individual: str) -> set:
        """
        busca regras que um individuo ta
        """
        if individual not in self.abox:
            return set() # nao achou individuo

        # procura nos diretos
        concepts_found = set(self.abox[individual])
        
        # bfs pra procurar as hierarquias nao diretas e enfileirar
        queue = collections.deque(self.abox[individual])
        
        while queue:
            current_concept = queue.popleft()
            
            # conceitos nao diretos
            super_concepts = self.tbox.get(current_concept, set())
            
            for super_c in super_concepts:
                # ve se esse conceito se liga a esse individuo
                if super_c not in concepts_found:
                    concepts_found.add(super_c) # adiciona na lista do individuo para buscas futuras
                    queue.append(super_c)      # coloca na lista pra buscas futuras
                    
        return concepts_found

    def check_instance(self, individual: str, concept_to_check: str) -> bool:
        """
        busca se individuo pertence a esse onceito
        """
        
        all_concepts_of_individual = self.get_all_concepts(individual)
        
        print(f"\n--- Verificando Instância: '{individual}' é um '{concept_to_check}'? ---")
        print(f"Raciocinador: Buscando todos os tipos de '{individual}'...")
        if not all_concepts_of_individual:
             print(f"Raciocinador: Indivíduo '{individual}' não encontrado no ABox.")
             return False
             
        print(f"Raciocinador: Tipos encontrados (diretos e inferidos) para '{individual}': {all_concepts_of_individual}")

        result = concept_to_check in all_concepts_of_individual
        
        if result:
            print(f"Resultado: SIM. '{concept_to_check}' está na lista de tipos inferidos.")
        else:
            print(f"Resultado: NÃO. '{concept_to_check}' não pôde ser inferido como um tipo para '{individual}'.")
            
        return result

# cria banco
kb = KnowledgeBase()

print("=== CONFIGURANDO O TBOX (REGRAS) ===")
kb.add_rule("Homem", "Mortal")
kb.add_rule("Homem", "Mamifero")
kb.add_rule("Gato", "Mamifero")
kb.add_rule("Mamifero", "Animal")
kb.add_rule("Peixe", "Animal")
print("-" * 30)

print("\n=== CONFIGURANDO O ABOX (FATOS) ===")
kb.add_fact("Socrates", "Homem")
kb.add_fact("Felix", "Gato")
kb.add_fact("Nemo", "Peixe")
print("-" * 30)


# valida resultados
resultado1 = kb.check_instance(individual="Socrates", concept_to_check="Homem")
resultado2 = kb.check_instance(individual="Socrates", concept_to_check="Mortal")
resultado3 = kb.check_instance(individual="Felix", concept_to_check="Animal")
resultado4 = kb.check_instance(individual="Felix", concept_to_check="Mortal")
resultado5 = kb.check_instance(individual="Nemo", concept_to_check="Mamifero")
