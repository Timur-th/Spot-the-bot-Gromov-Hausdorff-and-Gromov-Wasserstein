import numpy as np
import random
from scipy.optimize import linear_sum_assignment
import dgh as dgh

# Генерация искусственных метрических пространств
def generate_metric_space(n, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, 2)

# Вычисление матрицы расстояний (Евклидова метрика)
def compute_distance_matrix(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    return D

# Функция приспособленности (среднее отклонение, а не максимум)
def fitness(pi, D1, D2):
    n = len(pi)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = abs(D1[i, j] - D2[pi[i], pi[j]])
    return np.mean(cost_matrix)  # Вместо max берем среднее

# Орденный кроссовер (Order Crossover, OX)
def order_crossover(p1, p2):
    size = len(p1)
    cut1, cut2 = sorted(random.sample(range(size), 2))
    
    child = np.full(size, -1)
    child[cut1:cut2] = p1[cut1:cut2]

    pos = cut2
    for i in range(size):
        gene = p2[(cut2 + i) % size]
        if gene not in child:
            child[pos % size] = gene
            pos += 1
            
    return child

# Генетический алгоритм
def genetic_algorithm(D1, D2, pop_size=100, generations=300, mutation_rate=0.1):
    n = D1.shape[0]
    population = [np.random.permutation(n) for _ in range(pop_size)]
    
    for _ in range(generations):
        scores = sorted([(fitness(pi, D1, D2), pi) for pi in population], key=lambda x: x[0])
        population = [score[1] for score in scores[:pop_size // 2]]

        while len(population) < pop_size:
            p1, p2 = random.sample(population[:10], 2)
            child = order_crossover(p1, p2)
            population.append(child)

        for pi in population:
            if random.random() < mutation_rate:
                i, j = random.sample(range(n), 2)
                pi[i], pi[j] = pi[j], pi[i]

    return population[0]

# Tabu Search
def tabu_search(D1, D2, pi, iterations=50, tabu_size=10):
    best_pi = pi.copy()
    best_score = fitness(pi, D1, D2)
    tabu_list = set()
    
    for _ in range(iterations):
        best_neighbor = None
        best_neighbor_score = float("inf")
        
        for _ in range(20):  # Генерируем 20 соседей
            i, j = random.sample(range(len(pi)), 2)
            new_pi = pi.copy()
            new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
            
            if tuple(new_pi) not in tabu_list:
                score = fitness(new_pi, D1, D2)
                if score < best_neighbor_score:
                    best_neighbor = new_pi
                    best_neighbor_score = score
        
        if best_neighbor is None:
            break
        
        pi = best_neighbor
        tabu_list.add(tuple(pi))
        if len(tabu_list) > tabu_size:
            tabu_list.pop()
        
        if best_neighbor_score < best_score:
            best_pi, best_score = best_neighbor, best_neighbor_score
    
    return best_pi

# Основной код
def main():
    n = 200
    X = generate_metric_space(n, seed=42)
    Y = generate_metric_space(n, seed=24)
    
    D1 = compute_distance_matrix(X)
    D2 = compute_distance_matrix(Y)
    
    print("Запуск GA...")
    pi_ga = genetic_algorithm(D1, D2)
    print("GA завершен, dGH:", fitness(pi_ga, D1, D2))
    
    print("Оптимизация Tabu Search...")
    pi_tabu = tabu_search(D1, D2, pi_ga)
    print("Tabu завершен, dGH:", fitness(pi_tabu, D1, D2))

    print("dGH библиотечной функцией:", dgh.upper(D1, D2))

if __name__ == "__main__":
    main()