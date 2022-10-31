import numpy as np
from abc import ABC, abstractmethod
from numpy.lib.npyio import packbits

class Individual(ABC):
    def __init__(self, val=None, init_params=None):
        if val is not None:
            self.val = val
        else:
            self.val = self.random_init(init_params)

    @abstractmethod
    def paridade(self, other, paridade_params):
        pass

    @abstractmethod
    def mutacao(self, mutacao_params):
        pass

    @abstractmethod
    def random_init(self, init_params):
        pass

class Checkers(Individual):
    def paridade(self, other, pair_params):
        return Checkers(pair_params['alpha'] * self.val + (1 - pair_params['alpha']) * other.val)

    def mutacao(self, mutate_params):
        for _ in range(mutate_params['rate']):
            i, j = np.random.choice(range(len(self.value)), 2, replace=False)
            self.value[i], self.value[j] = self.value[j], self.value[i]

    def random_init(self, init_params):
        return np.random.choice(range(init_params['n_Enemy_Pecas']), init_params['n_Enemy_Pecas'], replace=False)

class Populacao:
    def __init__(self, size, fitness, individual_class, init_params):
        self.fitness = fitness
        self.individuals = [individual_class(init_params=init_params) for _ in range(size)]
        self.individuals.sort(key=lambda x: self.fitness(x))

    def substituicao(self, new_individuals):
        size = len(self.individuals)
        self.individuals.extend(new_individuals)
        self.individuals.sort(key=lambda x: self.fitness(x))
        self.individuals = self.individuals[-size:]

    def get_parents(self, n_filhos):
        maes = self.individuals[-2 * n_filhos::2]
        pais = self.individuals[-2 * n_filhos + 1::2]

        return maes, pais

class Evolucao:
    def __init__(self, pool_size, fitness, individual_class, n_filhos, pair_params, mutate_params, init_params):
        self.pair_params = pair_params
        self.mutate_params = mutate_params
        self.pool = Populacao(pool_size, fitness, individual_class, init_params)
        self.n_filhos = n_filhos

    def etapa(self):
        maes, pais = self.pool.get_parents(self.n_filhos)
        filhos = []

        for mae, pai in zip(maes, pais):
            filho = mae.paridade(pai, self.pair_params)
            filho.mutacao(self.mutate_params)
            filhos.append(filho)

        self.pool.substituicao(filhos)

def checkers_fitness_creator(EnemyPecas):
    matrix = []
    for i in EnemyPecas:
        row = []
        for j in EnemyPecas:
            row.append(np.linalg.norm(i - j))
        matrix.append(row)
    distances = np.array(matrix)

    def fitness(checkers):
        res = 0
        for i in range(len(checkers.value)):
            res += distances[checkers.value[i], checkers.value[(i + 1) % len(checkers.value)]]
        return -res

    return fitness

evo = Evolucao(
    pool_size=10, fitness=checkers_fitness_creator, individual_class=Checkers, n_filhos=7,
    pair_params={'alpha': 0.5},
    mutate_params={'rate': 0.25},
    init_params={'n_Enemy_Pecas': 12}
)
n_epocas = 100

for i in range(n_epocas):
    evo.etapa()

print(evo.pool.individuals[-1].val)