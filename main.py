import time
import threading
from solution import *
import matplotlib.pyplot as plt
from typing import Dict
from typing import List
import sys

def parse_input(io):
    """ simply parse i/o file.txt into a dict and return it """
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
    def __init__(self, population_size, puzzle: FutoshikiPuzzle):
        self.population_size = population_size
        self.puzzle = puzzle
        self.found_sol = False
        self.best_sol = None
        self.gen_scores = []
        self.elite_percent = 0.5
        self.mutate_prob = 0.1
        self.crossover_prob = 0.5

    def run(self):
        population = self.generate_random_population()
        gen = 0
        restart_counter = 0
        while not self.found_sol:
            population_to_score = self.eval_population(population)
            self.save_stats(population_to_score)
            if len(self.gen_scores) % 100 == 0:
                self.print_gen_stats()
            new_population = self.create_new_population(population_to_score)
            population = new_population
            gen += 1
            if not self.found_sol and gen % 2000 == 0:
                if restart_counter == 10:
                    print("Genetic Algo didn't find an optimal solution")
                    print("printing best solution so far..")
                    population_to_score = {k: v for k, v in
                                           sorted(population_to_score.items(), key=lambda item: item[1])}
                    self.best_sol = iter(population_to_score).__next__()
                    self.found_sol = True
                else:
                    population = self.restart()
                    restart_counter += 1



    def create_new_population(self, population_to_score: Dict[Solution, int]):
        """
        1. generate a weighted array
        2. copy elite population "as-is" to our new population
        3. generate child from crossover or replica of one parent with probability
        4. mutate child with probability
        :return: new population
        """
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

    def __create_weighted_population(self, population_to_score: Dict[Solution, int]) -> List[Solution]:
        """ returns a list of solutions where each solution appears *Score* times """
        weighted_population = []
        worst_score = self.puzzle.get_max_constraints()
        for solution in population_to_score.keys():
            score = population_to_score[solution]
            for j in range(worst_score - score):  # insert same solution with correlation to inverse best score
                weighted_population.append(solution)
        return weighted_population

    @abstractmethod
    def eval_population(self, population: List[Solution]):
        pass

    def plot_stat(self):
        avg = []
        best = []
        worst = []
        generations = []
        len_scores = len(self.gen_scores)
        # always show X bars
        X = 30
        num_bars = len_scores // X
        for i in range(0, len_scores, max(1, num_bars)):  # use max if len < X
            stats = self.gen_scores[i]
            best_score, worst_score, avg_score = stats
            best.append(best_score)
            worst.append(worst_score)
            avg.append(avg_score)
            generations.append(i + 1)


        width = len_scores * 0.01  # the width of the bars 1% of total len

        plt.bar(list(map(lambda x: x - width / 2, generations)), best, width, label='Best', color='#ff7f0e')
        plt.bar(list(map(lambda x: x + width / 2, generations)), worst, width, label='Worst', color='#1f77b4')
        plt.plot(generations, avg, '-or', ms=4, label='Avg')
        plt.ylabel('Scores')
        plt.xlabel('Generations')
        plt.title('Overview of performance')
        plt.legend()
        plt.show()

    def restart(self):
        """ performs a restart with new random solutions """
        return self.generate_random_population()

    def elitism(self, population_to_score: Dict[Solution, int]) -> List[Solution]:
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

    def generate_random_population(self) -> List[Solution]:
        return [self.puzzle.get_random_solution() for i in range(self.population_size)]

    def save_stats(self, population_to_score: Dict[Solution, int]):
        best_score, worst_score, avg_score = self.puzzle.get_best_worst_avg_score(population_to_score)
        self.gen_scores.append((best_score, worst_score, avg_score))

    def print_gen_stats(self):
        best_score, worst_score, avg_score = self.gen_scores[-1]
        print(
            f'Gen No. {len(self.gen_scores)}\tBest score {best_score}\tWorst score {worst_score}\tAvg score '
            f'{round(avg_score, 2)}'
        )


    def print_solution(self):
        self.puzzle.print_solution(self.best_sol)

class BasicGeneticAlgo(GeneticAlgo):
    def eval_population(self, population: List[Solution]):
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
    def eval_population(self, population: List[Solution]):
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
    def eval_population(self, population: List[Solution]):
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

def plot_compare(basic, lamarck, darwin):
    basic_scores, basic_name = basic
    lamarck_scores, lamarck_name = lamarck
    darwin_scores, darwin_name = darwin
    to_plot = [(get_gen_avg(basic_scores), basic_name),
               (get_gen_avg(lamarck_scores), lamarck_name),
               (get_gen_avg(darwin_scores), darwin_name)]
    to_plot = sorted(to_plot, key=lambda stat: len(stat[0][1]))
    for i in range(len(to_plot)):
        plt.plot(to_plot[i][0][1], to_plot[i][0][0], '-o', ms=1, label=f'Avg_{to_plot[i][1]}', zorder=len(to_plot)-i)

    plt.ylabel('Scores')
    plt.xlabel('Generations')
    plt.title('Different algorithms comparison Compare')
    plt.legend()
    plt.show()

def get_gen_avg(scores):
    generations = []
    avg = []
    for i, stats in enumerate(scores):
        best_score, worst_score, avg_score = stats
        avg.append(avg_score)
        generations.append(i + 1)
    return avg, generations

def execute_basic_alg(gen_num, puzzle):
    print("Executing Basic Algorithm\n")
    ga_basic = BasicGeneticAlgo(gen_num, puzzle)
    ga_basic.run()
    if ga_basic.best_sol:
        ga_basic.print_solution()
        ga_basic.plot_stat()

def execute_lamarck_alg(gen_num, puzzle):
    print("Executing Lamarck Algorithm")
    ga_lamarck = LamarckAlgo(gen_num, puzzle)
    ga_lamarck.run()
    if ga_lamarck.best_sol:
        ga_lamarck.print_solution()
        ga_lamarck.plot_stat()

def execute_darwin_alg(gen_num, puzzle):
    print("Executing Darwin Algorithm")
    ga_darwin = DarwinAlgo(gen_num, puzzle)
    ga_darwin.run()
    if ga_darwin.best_sol:
        ga_darwin.print_solution()
        ga_darwin.plot_stat()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Error: program expect one argument which is a path to a config file.txt")
    else:
        file_path = sys.argv[-1]

    gen_num = 100
    input_dict = parse_input(io=file_path)
    n = input_dict['N']
    msf = MatrixSolutionFactory()
    puzzle = FutoshikiPuzzle(n, input_dict['constraints'], input_dict['mat_init'], msf)
    constraints = input_dict['constraints']

    try:
        algorithm_selection = int(input("Enter a selection:\n1. Run all algorithms\n2. Run only one algorithm\n"))
        if algorithm_selection == 1:
            execute_basic_alg(gen_num, puzzle)
            execute_lamarck_alg(gen_num, puzzle)
            execute_darwin_alg(gen_num, puzzle)
        elif algorithm_selection == 2:
            specific_algorithm_selection = int(input("1. Basic\n2.Lamarck\n3.Darwin\n"))
            if specific_algorithm_selection == 1:
                execute_basic_alg(gen_num, puzzle)
            elif specific_algorithm_selection == 2:
                execute_lamarck_alg(gen_num, puzzle)
            elif specific_algorithm_selection == 3:
                execute_darwin_alg(gen_num, puzzle)
            else:
                raise ValueError
        else:
            raise ValueError
    except ValueError:
        sys.exit("Error: Please enter a valid selection")

