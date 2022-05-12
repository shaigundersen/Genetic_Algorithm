from solution import *
import matplotlib.pyplot as plt
UP = '^'
RIGHT = '>'
LEFT = '<'
SPACE = ' '



def parse_input(io):
    dic = {}
    with open(io, 'r') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines]  # remove all new-line
        i = 0
        dic['N'] = int(lines[i])
        i += 1
        mat_init_size = int(lines[i])
        i += 1
        dic['mat_init'] = []
        for j in range(mat_init_size):
            line = lines[i + j]
            splitted = line.split(' ')
            splitted = tuple([int(num) for num in splitted])
            dic['mat_init'].append(splitted)
        i += mat_init_size
        greater_init_size = int(lines[i])
        i += 1
        dic['constraints'] = []  # each constraint is a tuple of 2 (i,j) indexes, first is bigger than second
        for j in range(greater_init_size):
            line = lines[i + j]
            splitted = line.split(' ')
            splitted = [int(num) for num in splitted]
            big = (splitted[0], splitted[1])
            small = (splitted[2], splitted[3])
            dic['constraints'].append((big, small))

    return dic


class StrategySolver:
    def __init__(self):
        pass


class GeneticAlgo(ABC):
    def __init__(self, population_size, solution_factory: SolutionFactory, constraints):
        self.population_size = population_size
        self.population = [solution_factory.generate_solution() for i in range(population_size)]
        self.constraints = constraints
        self.found_sol = False
        self.best_sol = None

    def run(self):
        best_score =  5*4*2 + 2 + 8 + 1
        generations_scores = []
        gen = 1
        while not self.found_sol:
            # eval sols
            scores = self.eval_population()
            cur_best = min(scores)
            if cur_best < best_score:
                best_score = cur_best
            if gen % 100 == 0:
                print("------------------------------")
                print(f'gen -> {gen}')
                print(f'best score -> {best_score}')
                print(f'avg score -> {sum(scores)/len(scores)}')
            generations_scores.append(scores)
            weighted_population = self.create_weighted_population(scores)
            self.create_new_population(weighted_population, scores)
            gen += 1

    def create_new_population(self, weighted_population, scores):
        new_pop = []
        # elitism
        elite_size = int(self.population_size * 0.25)
        result = sorted(zip(self.population, scores), key=lambda tup: tup[1])
        for i in range(elite_size):
            new_pop.append(result[i][0])
        # elita = result[:elite_size]
        # new_pop += elita
        for i in range(self.population_size - elite_size):  # generate new population with fixed size
            parents = random.sample(weighted_population, 2)
            sol_A, sol_B = parents[0], parents[1]
            sol_A = sol_A.mutate(0.5)
            sol_B = sol_B.mutate(0.5)
            child_sol = sol_A.crossover(sol_B)
            new_pop.append(child_sol)
        self.population = new_pop

    @abstractmethod
    def optimize_elite(self, elite):
        pass
    def create_weighted_population(self, scores):
        weighted_population = []
        worst_score = 5*4*2 + 2 + 8
        for i, solution in enumerate(self.population):
            for j in range(worst_score - scores[i]):
                weighted_population.append(solution)
        return weighted_population

    def eval_population(self):
        scores = []
        for solution in self.population:
            score = solution.fitness(self.constraints)
            if score == 0:  # all constraints are met
                self.found_sol = True
                self.best_sol = copy.deepcopy(solution)
            scores.append(score)
        return scores


class LamarckAlgo(GeneticAlgo):
    def __init__(self, population_size, solution_factory: SolutionFactory, constraints, allow_opt=5):
        super().__init__(population_size, solution_factory, constraints)
        self.allow_opt = allow_opt

    def optimize_elite(self, elite):
        for i in range(self.allow_opt):
            if i < len(elite):
                elite[i].optimize()


if __name__ == '__main__':
    input_dict = parse_input(io='./input.txt')
    n = input_dict['N']
    msf = MatrixSolutionFactory(n, input_dict['mat_init'])
    constraints = input_dict['constraints']
    ga = GeneticAlgo(100, msf, constraints)
    ga.run()
    if ga.best_sol:
        ga.best_sol.print_solution()


