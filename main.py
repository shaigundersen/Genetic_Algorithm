import time

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

class GeneticAlgo(ABC):
    def __init__(self, population_size, puzzle: FPuzzle):
        self.population_size = population_size
        self.puzzle = puzzle
        self.found_sol = False
        self.best_sol = None
        self.gen_scores = []
        self.elite_percent = 0.35
        self.mutate_prob = 0.1
        self.crossover_prob = 0.5

    def run(self):
        population = self.generate_random_population()
        gen = 0
        while not self.found_sol:
            population_to_score = self.eval_population(population)
            self.save_stats(population_to_score)
            if len(self.gen_scores) % 100 == 0:
                self.print_gen_stats()
            new_population = self.create_new_population(population_to_score)
            population = new_population
            gen += 1

    def create_new_population(self, population_to_score: dict[Solution, int]):
        weighted_population = self.__create_weighted_population(population_to_score)
        new_pop = self.elitism(population_to_score)
        for i in range(self.population_size - len(new_pop)):
            p = random.random()
            # next child will be a result of either crossover between two parents or a copy of an individual
            if p > self.crossover_prob:
                parents = random.sample(weighted_population, 2)
                sol_A, sol_B = parents[0], parents[1]
                child_sol = sol_A.crossover(sol_B)
            else:
                child_sol = random.choice(weighted_population)
            p = random.random()
            # mutate next child with mutate_prob, else don't mutate
            if p > self.mutate_prob:
                child_sol = child_sol.mutate()
            new_pop.append(child_sol)
        return new_pop

    def __create_weighted_population(self, population_to_score: dict[Solution, int]) -> list[Solution]:
        """ returns a list of solutions where each solution appears *Score* times """
        weighted_population = []
        worst_score = self.puzzle.get_max_constraints()
        for solution in population_to_score.keys():
            score = population_to_score[solution]
            for j in range(worst_score - score):  # insert same solution with correlation to inverse best score
                weighted_population.append(solution)
        return weighted_population

    @abstractmethod
    def eval_population(self, population: list[Solution]):
        pass

    def plot_stat(self):
        avg = []
        best = []
        worst = []
        generations = []
        for i, stats in enumerate(self.gen_scores):
            best_score, worst_score, avg_score = stats
            best.append(best_score)
            worst.append(worst_score)
            avg.append(avg_score)
            generations.append(i + 1)

        width = 0.3  # the width of the bars

        fig, ax = plt.subplots()
        ax.bar(list(map(lambda x: x - width / 2, generations)), best, width, label='Best')
        ax.bar(list(map(lambda x: x + width / 2, generations)), worst, width, label='Worst')
        ax.plot(generations, avg, '-or', label='Avg')

        ax.set_ylabel('Score')
        ax.set_xlabel('Generation (X 1000)')
        ax.set_title('Overview of performance')
        if len(generations) > 10:  # prettify X axis labels
            generations = [generations[i] for i in range(0, len(generations), 2)]
        ax.set_xticks(generations)
        ax.legend()

        fig.tight_layout()

        plt.show()

    def restart(self):
        """ performs a restart with elite solutions and new random solutions """
        pass
        # scores = self.eval_population()
        # new_pop = self.elitism(scores)
        # for i in range(self.population_size - len(new_pop)):  # generate new population with fixed size
        #     new_pop.append(puzzle.get_random_solution())
        # self.population = new_pop
        # self.population = [puzzle.get_random_solution() for i in range(self.population_size)]

    def elitism(self, population_to_score: dict[Solution, int]) -> list[Solution]:
        """ returns *elite_percent* of the best solutions """
        population_to_score = {k: v for k, v in sorted(population_to_score.items(), key=lambda item: item[1])}
        elite_population = []
        elite_size = int(self.population_size * self.elite_percent)
        i = 0
        for solution in population_to_score.keys():
            if i == elite_size:
                break
            elite_population.append(solution)
            i += 1
        return elite_population

    def generate_random_population(self) -> list[Solution]:
        return [self.puzzle.get_random_solution() for i in range(self.population_size)]

    def save_stats(self, population_to_score: dict[Solution, int]):
        best_score, worst_score, avg_score = self.puzzle.get_best_worst_avg_score(population_to_score)
        self.gen_scores.append((best_score, worst_score, avg_score))

    def print_gen_stats(self):
        best_score, worst_score, avg_score = self.gen_scores[-1]
        print(
            f'Gen No. {len(self.gen_scores)}\tBest score {best_score}\tWorst score {worst_score}\tAvg score {avg_score}'
        )

class BasicGeneticAlgo(GeneticAlgo):
    def eval_population(self, population: list[Solution]):
        """ :returns a map of each solution to it's score """
        population_to_score = {}
        for solution in population:
            score = self.puzzle.fitness_func(solution)
            if self.puzzle.is_terminal_state(score):  # all constraints are met
                self.found_sol = True
                self.best_sol = copy.deepcopy(solution)
            population_to_score[solution] = score
        return population_to_score


class LamarckAlgo(GeneticAlgo):
    def eval_population(self, population: list[Solution]):
        """ :returns a map of each optimized solution to it's optimized score """
        population_to_score = {}
        for solution in population:
            optimized_solution = self.puzzle.optimization_func(solution)  # optimize before fitness
            optimized_score = self.puzzle.fitness_func(optimized_solution)
            if self.puzzle.is_terminal_state(optimized_score):  # all constraints are met
                self.found_sol = True
                self.best_sol = copy.deepcopy(optimized_solution)
            population_to_score[optimized_solution] = optimized_score
        return population_to_score


class DarwinAlgo(GeneticAlgo):
    def eval_population(self, population: list[Solution]):
        """ :returns a map of each solution to it's optimized score """
        population_to_score = {}
        for solution in population:
            score = self.puzzle.fitness_func(solution)  # so we know if this solution is a Terminal state
            optimized_solution = self.puzzle.optimization_func(solution)
            optimization_score = self.puzzle.fitness_func(optimized_solution)
            if self.puzzle.is_terminal_state(score):  # all constraints are met
                self.found_sol = True
                self.best_sol = copy.deepcopy(solution)
            population_to_score[solution] = optimization_score
        return population_to_score


if __name__ == '__main__':
    input_dict = parse_input(io='./5x5-easy.txt')
    n = input_dict['N']
    msf = MatrixSolutionFactory()
    puzzle = FPuzzle(n, input_dict['constraints'], input_dict['mat_init'], msf)
    constraints = input_dict['constraints']
    # ga = BasicGeneticAlgo(100, puzzle)
    ga = LamarckAlgo(100, puzzle)
    start = time.time()
    ga.run()
    end = time.time()
    print(f"GA finished in {end - start} seconds")
    if ga.best_sol:
        ga.best_sol.print_solution()
        ga.plot_stat()

