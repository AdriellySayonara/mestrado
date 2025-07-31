# ml_manager/genetic_algorithm.py
import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def run_ga_feature_selection(X_train, y_train, n_features):
    """
    Executa um Algoritmo Genético para encontrar o melhor subconjunto de features.
    """
    print(f"   Iniciando AG para selecionar features de um total de {n_features}.")

    # 1. SETUP DO ALGORITMO GENÉTICO (DEAP)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 2. FUNÇÃO DE APTIDÃO (FITNESS)
    def evaluate_features(individual):
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        if not selected_indices:
            return (0,)

        X_subset = X_train[:, selected_indices]
        
        # Usamos um classificador rápido com validação cruzada para uma estimativa robusta
        clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        score = np.mean(cross_val_score(clf, X_subset, y_train, cv=3, scoring='accuracy'))
        
        return (score,)

    # 3. OPERAÇÕES GENÉTICAS
    toolbox.register("evaluate", evaluate_features)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 4. EXECUÇÃO
    # Parâmetros do AG (menores para testes rápidos)
    POP_SIZE = 40
    NGEN = 50 # Número de gerações
    CXPB, MUTPB = 0.8, 0.2 # Probabilidades de crossover e mutação

    print(f"   ... Executando AG (População: {POP_SIZE}, Gerações: {NGEN})...")
    population = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, halloffame=hof, verbose=False)
    
    best_individual = hof[0]
    best_features_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
    
    print(f"   ... AG concluído. Melhor aptidão: {best_individual.fitness.values[0]:.4f}")
    print(f"   ... Features selecionadas: {len(best_features_indices)} de {n_features}")

    return best_features_indices