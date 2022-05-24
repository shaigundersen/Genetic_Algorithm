import random
import copy
from abc import ABC, abstractmethod
from typing import Dict

class Solution(ABC):
    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def crossover(self, solution: 'Solution'):
        pass

    # @abstractmethod
    # def print_solution(self):
    #     pass

    @abstractmethod
    def fitness(self, constraints, seeded_solution):
        pass

    @abstractmethod
    def optimize(self, greater_constraints, seeded_solution):
        pass

    @abstractmethod
    def get_sol(self):
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

    def optimize(self, greater_constraints, seeded_solution):
        """
         optimization swap places in row s.t -
         1. constraints are met
         2. seeded solution is met
         """
        sol_copy = copy.deepcopy(self)
        i = 0
        it = iter(greater_constraints)
        try:
            while i < sol_copy.__N:  # allowed up-to N optimization operations
                constraint = next(it)
                if not sol_copy.__greater_sign(constraint):
                    big, small = constraint
                    # only swap with same row, if we swap columns rows might get ruined and it will destroy inner logic
                    if big[0] == small[0]:
                        temp = sol_copy.__solution[big[0] - 1][big[1] - 1]
                        sol_copy.__solution[big[0] - 1][big[1] - 1] = sol_copy.__solution[small[0] - 1][small[1] - 1]
                        sol_copy.__solution[small[0] - 1][small[1] - 1] = temp
                        i += 1
        except StopIteration:
            for k in range(sol_copy.__N):
                for j in range(sol_copy.__N):
                    if i >= sol_copy.__N:
                        break
                    seed = seeded_solution.__solution[k][j]
                    if seed != 0 and sol_copy.__solution[k][j] != seed:
                        # swap places with where seed is supposed to be
                        idx = sol_copy.__solution[k].index(seed)
                        temp = sol_copy.__solution[k][j]
                        sol_copy.__solution[k][j] = sol_copy.__solution[k][idx]
                        sol_copy.__solution[k][idx] = temp
                        i += 1
                if i >= sol_copy.__N:
                    break
        return sol_copy

    def mutate(self):
        """
        picks a random row to mutate and swap two random elements
        :return copy of current Solution after swapping
        """
        sol_copy = copy.deepcopy(self)
        N = len(self.__solution)
        row = random.choice(range(N))
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
        :return copy of current Solution after swapping
        """
        sol_copy = copy.deepcopy(self)
        N = len(self.__solution)
        row = random.randint(0, N - 1)
        new_sol = self.__solution[:row]
        new_sol += solution.__solution[row:]
        sol_copy.__solution = new_sol
        # sol_copy._conform_seed()
        return sol_copy

    # def print_solution(self):
    #     N = len(self.__solution)
    #     print(2 * N * "*" + "*")
    #     for i in range(N):
    #         print("|", end='')
    #         for j in range(N):
    #             if j == N - 1:
    #                 print(self.__solution[i][j], end='|\n')
    #             else:
    #                 print(self.__solution[i][j], end=' ')
    #     print(2 * N * "*" + "*")


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

    def get_sol(self):
        return self.__solution

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


class FutoshikiPuzzle:
    def __init__(self, N, greater_constraints, seed_constraints, factory: SolutionFactory):
        self.__board_size = N
        self.__greater_constraints = greater_constraints
        self.__seed_constraints = seed_constraints
        self.__factory = factory
        self.__seeded_solution = factory.generate_seeded_solution(self.__board_size, self.__seed_constraints)

    def get_max_constraints(self):
        """ :returns num of constraints to be met.
         i.e each row and column is a permutation + initial board values + greater signs """
        return self.__board_size * (self.__board_size - 1) * 2 \
               + len(self.__greater_constraints) + len(self.__seed_constraints)

    def fitness_func(self, solution: Solution):
        return solution.fitness(self.__greater_constraints, self.__seeded_solution)

    def optimization_func(self, solution: Solution):
        return solution.optimize(self.__greater_constraints, self.__seeded_solution)

    def is_terminal_state(self, score):
        return score == 0

    def get_random_solution(self) -> Solution:
        return self.__factory.generate_random_solution(self.__board_size)

    def get_best_worst_avg_score(self, population_to_score: Dict[Solution, int]):
        best_score = self.get_max_constraints()  # being best means close to 0
        worst_score = 0  # being worst means close to num constraints
        sum = 0
        for score in population_to_score.values():
            if score < best_score:
                best_score = score
            if score > worst_score:
                worst_score = score
            sum += score
        return best_score, worst_score, sum / len(population_to_score)

    def print_solution(self, solution: Solution):
        N = len(solution.get_sol())
        solution_to_print = solution.get_sol()
        print(7 * N * "*" + "*")
        for i in range(N):
            print("|", end='')
            for j in range(N):
                if j == N - 1:
                    print(str(solution_to_print[i][j]) + " ", end='|\n')
                else:
                    if ((i+1,j+2),(i+1,j+1)) in self.__greater_constraints:
                        print(str(solution_to_print[i][j]) + " | < |", end=' ')
                    elif ((i+1,j+1),(i+1,j+2)) in self.__greater_constraints:
                        print(str(solution_to_print[i][j]) + " | > |", end=' ')
                    else:
                        print(str(solution_to_print[i][j]) + " |   |", end=' ')
            if not i == N-1:
                print("--" * 4 * (N-1)+"-"*4)

                print("--" * 4 * (N-1)+"-"*4)
        print(7 * N * "*" + "*")