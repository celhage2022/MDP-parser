import antlr4
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import random

class Model:
    def __init__(self):
        self.states = set()
        self.actions = set()
        self.transitions = {}
        self.dico_etats_actions = {}
        self.matrices_transitions = {}
        self.matrice_transition_sans_action = {}

    def remplissage_dico_etats_actions(self):
        # Parcourir tous les états du modèle
        for state in self.states:
            actions_from_state = []  # Initialiser un ensemble vide pour les actions de cet état
            # Parcourir toutes les transitions sortantes de cet état
            for action in self.transitions.get(state, {}).keys():
                if action is not None and action in self.actions :  # Ignorer les transitions sans action
                    actions_from_state.append(action)
            # Ajouter les actions disponibles pour cet état dans le dictionnaire
            self.dico_etats_actions[state] = actions_from_state
    
    def remplissage_matrices_transitions(self):
        # Parcourir toutes les actions disponibles dans le modèle
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
    model = Model()
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
    # etat = next(iter(model.transitions.values()))
    etat = 'S0'
    try: 
        while True:
    
            if None in model.transitions[etat] :
                
                print(f'Etat actuel = {etat}')
                somme, somme_proba = s_proba(model.transitions[etat][None])
                presentation_suite(etat, somme, model.transitions[etat][None])
                etat = choix_prochaine_etat(somme_proba)
            
            else :   
                print(f'Etat actuel = {etat}')        
                actions_possibles = list(model.transitions[etat].keys())
                choix = input(f'choisir parmi {actions_possibles} : ')

                if choix == 'exit':
                    break

                while choix not in actions_possibles:
                    choix = input(f"{choix} n'est pas dans {actions_possibles}" )
                
                somme, somme_proba = s_proba(model.transitions[etat][choix])
                presentation_suite(etat, somme, model.transitions[etat][choix])
                etat = choix_prochaine_etat(somme_proba)
    except EOFError as e:
        print('EOFError')

    print('Merci et au revoir')



def s_proba(dic):
    somme = 0
    somme_proba = {}
    for e in dic.keys() :
        somme += dic[e]
        somme_proba[e] = somme

    for cle, valeur in somme_proba.items():
        somme_proba[cle] = valeur / somme
    return(somme, somme_proba)


def presentation_suite(etat, somme, dic):
    print('Les possibles prochaines états sont')
    for e in dic.keys() : 
        valeur = dic[e]
        print(f'{e} avec une proba {valeur/somme}')
    return()


def choix_prochaine_etat(somme_proba):
    aleatoire  = random.random()
    for cle, valeur in somme_proba.items() :
        if valeur > aleatoire : 
            return(cle)

        

if __name__ == '__main__':
    main()