import antlr4
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import random
import networkx as nx
import matplotlib.pyplot as plt


class MDP:
    def __init__(self):
        self.current_state = None
        self.states = set()
        self.actions = set()
        self.transitions = {}
        self.dico_etats_actions = {}
        self.matrices_transitions = {}
        self.matrice_transition_sans_action = {}

    def remplissage_dico_etats_actions(self):
        for state in self.states:
            actions_from_state = []  # Initialiser un ensemble vide pour les actions de cet état
            # Parcourir toutes les transitions sortantes de cet état
            for action in self.transitions.get(state, {}).keys():
                if action is not None and action in self.actions :  # Ignorer les transitions sans action
                    actions_from_state.append(action)
            # Ajouter les actions disponibles pour cet état dans le dictionnaire
            self.dico_etats_actions[state] = actions_from_state
    
    def remplissage_matrices_transitions(self):
        for action in self.actions:
            # Créer une matrice de transition pour cette action
            transition_matrix = {}
            # Parcourir tous les états du modèle
            for source_state in self.states:
                # Initialiser une ligne de la matrice pour cet état source
                transition_matrix[source_state] = {}
                # Récupérer les transitions sortantes de cet état pour cette action
                transitions = self.transitions.get(source_state, {}).get(action, {})
                # Parcourir tous les états du modèle (y compris l'état source) pour remplir la ligne de la matrice
                for target_state in self.states:
                    # Remplir la probabilité de transition de l'état source à l'état cible
                    transition_matrix[source_state][target_state] = transitions.get(target_state, 0)
            # Ajouter la matrice de transition associée à l'action dans le dictionnaire matrices_transitions
            self.matrices_transitions[action] = transition_matrix

        no_action_transition_matrix = {}
        for source_state in self.states:
            no_action_transition_matrix[source_state] = {}
            # Récupérer les transitions sortantes de cet état sans action
            transitions = self.transitions.get(source_state, {}).get(None, {})
            for target_state in self.states:
                no_action_transition_matrix[source_state][target_state] = transitions.get(target_state, 0)
        # Ajouter la matrice de transition pour les états sans action dans le dictionnaire matrices_transitions
        self.matrice_transition_sans_action = no_action_transition_matrix


    def presentation_suite(self):
        '''
        Presente à l'utilisateur la suite dans le MDP
        '''
        continuer = input('Continuer ? [y/n] :')
        if continuer == 'y':
            continuer = True 
            if None in self.transitions[self.current_state]: 
                print('vous êtes sur un choix probabiliste,\n appuyez sur Entrée pour continuer :')
            else : 
                display = []
                for e in self.transition[self.current_state] :
                    display.append(e) 

                print(f'Veuillez faire un choix dans : {display}')
        elif continuer == 'n' : 
            continuer = False
        return(continuer)
  
    
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
        somme = 0
        somme_proba = {}
        for e in self.transitons[self.current_state][a].keys() :
            somme += self.transitons[self.current_state][a][e]
            somme_proba[e] = somme

        for cle, valeur in somme_proba.items():
            somme_proba[cle] = valeur / somme
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
        somme_proba = self.s_proba(a)
        self.current_state = self.prochain_etat(somme_proba)
        return()            

    def plot_graph(self):
        G = nx.DiGraph

        list_node = []
        list_choix = []

        for etat in self.transitions :
            G.add_node(etat, label = etat)
            if etat != self.current_state :
                list_node.append(etat)
        
        for etat in self.transitions :
            if None in self.transitions :
                somme = self.somme(self.transitions[etat][None])
                for node in self.transitions[etat][None].keys():
                    G.add_edge(etat, node, 
                               label = str(self.transitions[etat][None][node]/somme) )
            else :
                for choix in self.transitions[etat]:
                    G.add_node(etat+choix, label = choix)
                    list_choix.append(str(etat+choix))

                    G.add_edge(etat, etat+choix)
                    somme = self.somme(self.transitions[etat][choix])
                    for p_etat in self.transitions[etat][choix].keys():
                        G.add_edge(etat+choix, p_etat,
                                   label = str(self.transitions[etat][choix]/somme) )
                        
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color='red', nodelist=[self.current_state], node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, node_color='blue', nodelist=list_node, node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, node_color='gray', nodelist=list_choix, node_size=250, alpha=0.8)

        nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes})
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', connectionstyle="arc3,rad=0.1", arrowstyle='-|>')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): G[i][j]['label'] for i, j in G.edges})
        plt.axis('off')
        plt.show()

    def plot_graph():
        G = nx.DiGraph()

        list_node = []
        list_choix = []

        for etat in transitions:
            G.add_node(etat, label = etat)  
            if etat != current_state:
                list_node.append(etat)

        for etat in transitions:
            if None in transitions[etat]:
                total_prob = sum(transitions[etat][None].values()) 
                for node, prob in transitions[etat][None].items():
                    G.add_edge(etat, node, label=str(prob / total_prob))
            else:
                for choix, next_states in transitions[etat].items():
                    G.add_node(etat + choix, label=choix)
                    list_choix.append(etat + choix)

                    G.add_edge(etat, etat + choix)
                    total_prob = sum(next_states.values())
                    for p_etat, prob in next_states.items():
                        G.add_edge(etat + choix, p_etat, label=str(prob / total_prob))

        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color='red', nodelist=[current_state], node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, node_color='blue', nodelist=list_node, node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, node_color='gray', nodelist=list_choix, node_size=250, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='black', connectionstyle="arc3,rad=0.15", arrowstyle='-|>')
        
        nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes})
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): G[i][j]['label'] for i, j in G.edges if 'label' in G[i][j]})

        plt.axis('off')
        plt.show()
        return()


class gramPrintListener(gramListener):
    def __init__(self, model):
        self.model = model

    def enterDefstates(self, ctx):
        for state in ctx.ID():
            self.model.states.add(str(state))

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

        

def main():
    model = MDP()
    lexer = gramLexer(antlr4.StdinStream())
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
    model.remplissage_dico_etats_actions()
    print("Dictionnaire etats actions:", model.dico_etats_actions)
    model.remplissage_matrices_transitions()
    print("Matrices transition :", model.matrices_transitions)
    print("Matrice transition sans action :", model.matrice_transition_sans_action)

    #depart
    model.current_state = 'S0' 
    continuer = True
    while continuer :
        continuer = model.presentation_suite()
        a = input()
        if a ==  '':
            model.avance()
        else :
            model.avance(a)
        
        model.plot_graph()

    print('Merci et au revoir')

        
if __name__ == '__main__':
    main()