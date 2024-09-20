import random

import numpy as np
 
# Função objetivo

def f(x):

    return x**3 - 6*x + 14
 
# Função para decodificar um vetor binário em um número real na faixa [-10, 10]

def decode(binary_vector, lower_bound, upper_bound):

    binary_string = ''.join(str(bit) for bit in binary_vector)

    decimal_value = int(binary_string, 2)

    max_decimal = 2**len(binary_vector) - 1

    return lower_bound + (upper_bound - lower_bound) * decimal_value / max_decimal
 
# Função para calcular a aptidão (fitness) de um indivíduo

def fitness(individual):

    x = decode(individual, -10, 10)

    return -f(x)  # Negativo porque estamos minimizando
 
# Função para criar uma população inicial

def create_population(pop_size, chromosome_length):

    return [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(pop_size)]
 
# Função para seleção por torneio

def tournament_selection(population, fitnesses, k=3):

    selected = random.sample(list(zip(population, fitnesses)), k)

    selected.sort(key=lambda x: x[1], reverse=True)

    return selected[0][0]
 
# Função para crossover de 1 ou 2 pontos

def crossover(parent1, parent2, num_points=1):

    if num_points == 1:

        point = random.randint(1, len(parent1) - 1)

        return parent1[:point] + parent2[point:]

    elif num_points == 2:

        point1 = random.randint(1, len(parent1) - 2)

        point2 = random.randint(point1 + 1, len(parent1) - 1)

        return parent1[:point1] + parent2[point1:point2] + parent1[point2:]
 
# Função para mutação

def mutate(individual, mutation_rate=0.01):

    for i in range(len(individual)):

        if random.random() < mutation_rate:

            individual[i] = 1 - individual[i]
 
# Função principal do algoritmo genético

def genetic_algorithm(pop_size=10, chromosome_length=16, generations=100, mutation_rate=0.01, crossover_points=1, elitism=True, elite_percent=0.1):

    population = create_population(pop_size, chromosome_length)

    best_solutions = []
 
    for generation in range(generations):

        fitnesses = [fitness(individual) for individual in population]

        new_population = []
 
        if elitism:

            num_elites = int(elite_percent * pop_size)

            elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:num_elites]

            new_population.extend([elite[0] for elite in elites])
 
        while len(new_population) < pop_size:

            parent1 = tournament_selection(population, fitnesses)

            parent2 = tournament_selection(population, fitnesses)

            child = crossover(parent1, parent2, crossover_points)

            mutate(child, mutation_rate)

            new_population.append(child)
 
        population = new_population

        best_solution = max(population, key=lambda x: fitness(x))

        best_solutions.append((decode(best_solution, -10, 10), -fitness(best_solution)))
 
    return best_solutions
 
# Executar o algoritmo genético

resultados = genetic_algorithm(pop_size=10, generations=50, mutation_rate=0.01, crossover_points=2, elitism=True, elite_percent=0.1)
 
# Exibir os resultados

for valor, solucao in resultados:

    print(f"Valor de x: {valor}, Valor mínimo de f(x): {solucao}")

 