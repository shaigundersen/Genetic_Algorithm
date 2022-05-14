import random
import copy
from abc import ABC, abstractmethod
import math

class Solution(ABC):
    @abstractmethod
    def mutate(self, percent):
        pass

    @abstractmethod
    def crossover(self, solution: 'Solution'):
        pass

    @abstractmethod
    def print_solution(self):
        pass

    @abstractmethod
    def fitness(self, constraints, seeded_solution):
        pass



class MatrixSolution(Solution):
    def __init__(self, N):
        self.__N = N
        self.__solution = []
        base_permutation = list(range(1, N + 1))  # create list of [1,2,..,N]
        for i in range(N):
            base_permutation_copy = copy.deepcopy(base_permutation)
            random.shuffle(base_permutation_copy)
            self.__solution.append(base_permutation_copy)

    def mutate(self, percent):
        """
        picks a percent of rows to mutate
        for each picked row, swap two random elements
        :return copy of current Solution after swapping
        """
        sol_copy = copy.deepcopy(self)
        N = len(self.__solution)
        rows_to_mutate = math.ceil(N * percent)
        rows = random.sample(range(N), rows_to_mutate)
        for row in rows:
            cols = random.sample(range(N), 2)  # pick two indexes from that row to be swapped
            temp = sol_copy.__solution[row][cols[0]]
            sol_copy.__solution[row][cols[0]] = sol_copy.__solution[row][cols[1]]
            sol_copy.__solution[row][cols[1]] = temp
        # sol_copy._conform_seed()
        return sol_copy

    def crossover(self, solution: 'MatrixSolution'):
        """
        picks a random row - R.
        Append to grid3 all rows from grid1 until R and all rows from grid2 from R until the end
        """
        sol_copy = copy.deepcopy(self)
        N = len(self.__solution)
        row = random.randint(0, N - 1)
        new_sol = self.__solution[:row]
        new_sol += solution.__solution[row:]
        sol_copy.__solution = new_sol
        # sol_copy._conform_seed()
        return sol_copy

    def print_solution(self):
        N = len(self.__solution)
        print(2 * N * "*" + "*")
        for i in range(N):
            print("|", end='')
            for j in range(N):
                if j == N - 1:
                    print(self.__solution[i][j], end='|\n')
                else:
                    print(self.__solution[i][j], end=' ')
        print(2 * N * "*" + "*")

    def __consistent(self, solution):
        """ returns num of diff elements in each row. Best case is 0 when each row is a permutation """
        return sum(self.__N - len(set(row)) for row in solution)

    def __compare_to_seed(self, seeded_solution: 'MatrixSolution'):
        """ count number of mismatches between initial seeded solution with current solution """
        count = 0
        for seed_row, row in zip(seeded_solution.__solution, self.__solution):
            for x_sr, x_r in zip(seed_row, row):
                if x_sr:  # when x_sr == 0 -> no seed
                    if x_sr != x_r:  # mismatch
                        count += 1
        return count

    def fitness(self, greater_constraints, seeded_solution):
        # calc diff elements in each ROW
        fit = self.__consistent(self.__solution)
        # calc diff elements in each COL
        fit += self.__consistent(zip(*self.__solution))
        # calc diff from seed (given digits)
        fit += self.__compare_to_seed(seeded_solution)
        # calc diff greater sign
        fit += self._eval_greater_constraints(greater_constraints)
        return fit

    def _eval_greater_constraints(self, constraints):
        """ returns number of satisfied constraints """
        count = 0
        for constraint in constraints:
            if not self.__greater_sign(constraint):
                count += 1
        return count

    def __greater_sign(self, constraint):
        big, small = constraint
        return self.__solution[big[0] - 1][big[1] - 1] > self.__solution[small[0] - 1][small[1] - 1]

    @classmethod
    def get_seeded_solution(cls, N, seeds):
        sol = cls(N)
        sol.__solution = [[0 for j in range(N)] for i in range(N)]
        for seed in seeds:
            i, j, num = seed
            sol.__solution[i - 1][j - 1] = num
        return sol


class SolutionFactory(ABC):
    @abstractmethod
    def generate_random_solution(self, N) -> Solution:
        pass
    @abstractmethod
    def generate_seeded_solution(self, N, seeds) -> Solution:
        pass


class MatrixSolutionFactory(SolutionFactory):

    def generate_random_solution(self, N) -> MatrixSolution:
        return MatrixSolution(N)

    def generate_seeded_solution(self, N, seeds) -> MatrixSolution:
        return MatrixSolution.get_seeded_solution(N, seeds)


class FPuzzle:
    def __init__(self, N, greater_constraints, seed_constraints, factory: SolutionFactory):
        self.__board_size = N
        self.__greater_constraints = greater_constraints
        self.__seed_constraints = seed_constraints
        self.__factory = factory
        self.__seeded_solution = factory.generate_seeded_solution(self.__board_size, self.__seed_constraints)

    def get_size(self):
        return self.__board_size

    def get_max_constraints(self):
        return self.__board_size * (self.__board_size - 1) * 2 \
               + len(self.__greater_constraints) + len(self.__seed_constraints)

    def fitness_func(self, solution: Solution):
        return solution.fitness(self.__greater_constraints, self.__seeded_solution)

    def is_terminal_state(self, score):
        return score == 0

    def get_random_solution(self) -> Solution:
        return self.__factory.generate_random_solution(self.__board_size)
