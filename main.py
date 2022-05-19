import time
import threading
from solution import *
import matplotlib.pyplot as plt



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
            if not self.found_sol and gen % 1000 == 0:
                population = self.restart()

    def create_new_population(self, population_to_score: dict[Solution, int]):
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


def plot_compare2(lamarck, darwin):  # todo REMOVE when DONE
    lamarck_scores, lamarck_name = lamarck
    darwin_scores, darwin_name = darwin
    to_plot = [(get_gen_avg(lamarck_scores), lamarck_name), (get_gen_avg(darwin_scores), darwin_name)]
    to_plot = sorted(to_plot, key=lambda stat: len(stat[0][1]))
    for i in range(len(to_plot)):
        plt.plot(to_plot[i][0][1], to_plot[i][0][0], '-o', ms=1, label=f'Avg_{to_plot[i][1]}', zorder=len(to_plot)-i)

    plt.ylabel('Scores')
    plt.xlabel('Generations')
    plt.title('Different algorithms comparison Compare')
    plt.legend()
    plt.show()

def p():  # todo REMOVE when DONE
    """ trying out graph plotting """
    best = []
    worst = []
    avg = []
    generations = []
    r = 30
    width = r*0.01
    # always show 30 bars
    for i in range(0, r, max(1, r // 30)):
        best.append(20)
        worst.append(2)
        avg.append(10)
        generations.append(i + 1)

    plt.bar(list(map(lambda x: x - width / 2, generations)), best, width, label='Best', color='#ff7f0e')
    plt.bar(list(map(lambda x: x + width / 2, generations)), worst, width, label='Worst', color='#1f77b4')
    plt.plot(generations, avg, '-,r', ms=4, label='Avg')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # p()

    gen_num = 100
    input_dict = parse_input(io='./5x5-easy.txt')
    n = input_dict['N']
    msf = MatrixSolutionFactory()
    puzzle = FutoshikiPuzzle(n, input_dict['constraints'], input_dict['mat_init'], msf)
    constraints = input_dict['constraints']

    # ga = BasicGeneticAlgo(gen_num, puzzle)
    # ga = LamarckAlgo(gen_num, puzzle)
    # ga = DarwinAlgo(gen_num, puzzle)
    # start = time.time()
    # ga.run()
    # end = time.time()
    # print(f"GA finished in {end - start} seconds")
    # if ga.best_sol:
    #     ga.best_sol.print_solution()
    #     ga.plot_stat()

    ###### for plotting ########
    # algos = [BasicGeneticAlgo(gen_num, puzzle), LamarckAlgo(gen_num, puzzle), DarwinAlgo(gen_num, puzzle)]
    algos = [LamarckAlgo(gen_num, puzzle), DarwinAlgo(gen_num, puzzle)]
    threads = []
    for alg in algos:
        t = threading.Thread(target=alg.run)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # plot_compare(basic=(algos[0].gen_scores, "Basic"),lamarck=(algos[1].gen_scores, "Lamark"), darwin=(algos[2].gen_scores, "Darwin"))
    plot_compare2(lamarck=(algos[0].gen_scores, "Lamark"), darwin=(algos[1].gen_scores, "Darwin"))
