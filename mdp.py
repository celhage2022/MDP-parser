import antlr4
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import random
import networkx as nx
import matplotlib.pyplot as plt
import click


class MDP:  
    def __init__(self):
        self.current_state = None
        self.states = {}
        self.actions = set()
        self.transitions = {}
        self.pos = None


    def presentation_suite(self):
        '''
        Presente à l'utilisateur la suite dans le MDP
        '''
        if None in self.transitions[self.current_state]: 
            print('Vous êtes sur un choix probabiliste. Appuyez sur Entrée pour continuer :')
        else : 
            display = []
            for e in self.transitions[self.current_state] :
                display.append(e) 

            print(f'Veuillez faire un choix dans : {display}')
        return()
  
    
    def s_proba(self, a = None):
        '''
        Parameters
        ----------

        Returns
        ----------
        somme : int 
            Somme des poids des etats possibles suivants
        somme_proba : dict
            Dictionnaire, la clé est un état et le contenu est la sa proba au tour suivant additionné à 
            celle de l'etat avant lui dans le dictionnaire.
        '''
        while a not in self.transitions[self.current_state]:
            print(f"{a} n'est pas dans {list(self.transitions[self.current_state].keys())}")
            a = input(f'choisir vraiment dans {list(self.transitions[self.current_state].keys())}')
        somme = 0
        somme_proba = {}
        for e in self.transitions[self.current_state][a].keys() :
            somme += self.transitions[self.current_state][a][e]
            somme_proba[e] = somme

        for cle in somme_proba.keys():
            somme_proba[cle] = somme_proba[cle] / somme
        return(somme, somme_proba)


    def prochain_etat(self, somme_proba):
        '''
        Parameters
        ----------
        somme_proba : dict
            Dictionnaire, la clé est un état et le contenu est la sa proba au tour suivant additionné à 
            celle de l'etat avant lui dans le dictionnaire.

        Returns
        ----------
        cle : str
            Prochain etat
        '''
        aleatoire  = random.random()
        for cle, valeur in somme_proba.items() :
            if valeur > aleatoire : 
                return(cle)
    
    
    def avance(self, a = None):
        '''
        Fais un pas dans le MDP
        '''
        _, somme_proba = self.s_proba(a)
        self.current_state = self.prochain_etat(somme_proba)
        return()            


    def plot_graph(self):
        G = nx.DiGraph()

        list_node = []
        list_choix = []

        for etat in self.transitions:
            G.add_node(etat, label = etat)  
            if etat != self.current_state:
                list_node.append(etat)

        for etat in self.transitions:
            if None in self.transitions[etat]:
                total_prob = sum(self.transitions[etat][None].values()) 
                for node, prob in self.transitions[etat][None].items():
                    G.add_edge(etat, node, label=str(prob / total_prob))
            else:
                for choix, next_states in self.transitions[etat].items():
                    G.add_node(etat + choix, label=choix)
                    list_choix.append(etat + choix)

                    G.add_edge(etat, etat + choix)
                    total_prob = sum(next_states.values())
                    for p_etat, prob in next_states.items():
                        G.add_edge(etat + choix, p_etat, label=str(prob / total_prob))

        if self.pos is None :
            self.pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, self.pos, node_color='red', nodelist=[self.current_state], node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, self.pos, node_color='blue', nodelist=list_node, node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, self.pos, node_color='gray', nodelist=list_choix, node_size=250, alpha=0.8)
        nx.draw_networkx_edges(G, self.pos, width=1, edge_color='black', connectionstyle="arc3,rad=0.15", arrowstyle='-|>')
        
        nx.draw_networkx_labels(G, self.pos, labels={i: G.nodes[i]['label'] for i in G.nodes})
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels={(i, j): G[i][j]['label'] for i, j in G.edges if 'label' in G[i][j]})

        plt.axis('off')
        plt.show()
        return()
    
    def verif_model(self):
        '''
        Verifie que les états et les actions sont bien déclarés dans le préambule. 
        Verifie aussi qu'on ne melange pas MDP et MC
        '''
        for etat in self.transitions:
            if etat not in self.states:
                raise ValueError(f"{etat} n'est pas dans {self.states}, le fichier input comporte une erreur")

        for etat in self.transitions:
            for action in self.transitions[etat]:
                if action is not None :
                    if action not in self.actions:
                        raise ValueError(f"{action} n'est pas dans {self.actions}, le fichier input comporte une erreur")
            
        for etat in self.transitions:
            for action in self.transitions[etat]:
                for cle in self.transitions[etat][action].keys():
                    if cle not in self.states:
                        raise ValueError(f"{cle} n'est pas dans {self.states}, le fichier input comporte une erreur")
        
        for etat in self.transitions:
            if None in self.transitions[etat]:
                if len(self.transitions[etat]) > 1:
                    raise ValueError(f"Il y a un melange entre MC et MDP dans l'etat {etat}, le fichier input comporte une erreur")
                
    

class gramPrintListener(gramListener):
    def __init__(self, model):
        self.model = model

    def enterDefstates(self, ctx):
        for state_def in ctx.stateDef():
            state_name = str(state_def.ID())
            state_value = int(state_def.INT().getText())
            self.model.states[state_name] = state_value

    def enterDefactions(self, ctx):
        for action in ctx.ID():
            self.model.actions.add(str(action))

    def enterTransact(self, ctx):
        source_state = str(ctx.ID(0))
        action = str(ctx.ID(1))
        target_states = [str(x) for x in ctx.ID()[2:]]
        weights = [int(x.getText()) for x in ctx.INT()]
        self.model.transitions.setdefault(source_state, {}).setdefault(action, {})
        for target_state, weight in zip(target_states, weights):
            self.model.transitions[source_state][action][target_state] = weight

    def enterTransnoact(self, ctx):
        source_state = str(ctx.ID(0))
        target_states = [str(x) for x in ctx.ID()[1:]]
        weights = [int(x.getText()) for x in ctx.INT()]
        self.model.transitions.setdefault(source_state, {}).setdefault(None, {})
        for target_state, weight in zip(target_states, weights):
            self.model.transitions[source_state][None][target_state] = weight



def parse_file(file_content):
    model = MDP()
    lexer = gramLexer(antlr4.InputStream(file_content))
    stream = antlr4.CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener(model)
    walker = antlr4.ParseTreeWalker()
    walker.walk(printer, tree)

    # Afficher les informations collectées
    print("States:", model.states)
    print("Actions:", model.actions)
    print("Transitions:", model.transitions)

    model.verif_model()

    model.current_state = click.prompt(f'choisir un etat de depart dans {list(model.states.keys())}', type=str)
    while model.current_state not in model.states:
        print(f"{model.current_state} n'est pas dans {list(model.states.keys())}")
        model.current_state = click.prompt(f'choisir un etat de depart vraiment dans {list(model.states.keys())}', type=str)

    nbr_tour = click.prompt('Combien de tour voulez vous faire ? ', type=int)

    model.plot_graph()
    for _ in range(nbr_tour):
        model.presentation_suite()
        a = input()
        if a == '':
            model.avance()
        else:
            model.avance(a)

        model.plot_graph()

    print('Merci et au revoir')


if __name__ == '__main__':
    file_name = click.prompt('Nom du fichier input ? ', type=str)
    with open(file_name, 'r') as file:
        file_content = file.read()
        parse_file(file_content)