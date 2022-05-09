import random

UP = '^'
RIGHT = '>'
LEFT = '<'
SPACE = ' '

def print_grid(mat, constraints):
    l = len(mat)
    print(2 * l * "*" + "*")
    for i in range(len(mat)):
        print("|", end='')
        for j in range(len(mat[i])):
            # for big, small in constraints:
            #     b_i, b_j = big
            #     s_i, s_j = small

            if j == len(mat[i]) - 1:
                print(mat[i][j], end='|\n')
            else:
                print(mat[i][j], end=' ')
    print(2 * l * "*" + "*")

def parse_input(io):
    dic = {'N': 3}
    return dic

def make_grid(N):
    g = []
    for i in range(N + N - 1):
        # even row
        if i % 2 == 0:
            row = []
            j = 0
            while j < N - 1:
                row.append(j)
                r = random.randint(0, 2)
                if r % 2 == 0:
                    row.append(LEFT)
                else:
                    row.append(SPACE)
                j += 1
            row.append(j)
        else:
            # odd row
            row = []
            col = 0
            while col < N:
                if col % 2 == 0:
                    row.append(UP)
                else:
                    row.append(SPACE)
                col += 1
            row.append(SPACE)


        g.append(row)
    return g

def make_grid2(N):
    c = 0
    g = []
    for i in range(game_dict['N']):
        row = []
        for j in range(game_dict['N']):
            row.append(c)
            c += 1
        g.append(row)
    return g


def get_score(grid):
    pass

def eval_function(grid, constraints):
    count = 0
    for big, small in constraints:
        if not (grid[big[0] - 1][big[1] - 1] > grid[small[0] - 1][small[1] - 1]):
            count += 1
    return count

def mutate(grid, N):
    """ returns a new grid where two elements from a random row are swapped """
    import copy
    temp_grid = copy.deepcopy(grid)
    row = random.randint(0, N - 1)
    index = random.sample(range(N), 2)
    temp = temp_grid[row][index[0]]
    temp_grid[row][index[0]] = temp_grid[row][index[1]]
    temp_grid[row][index[1]] = temp
    return temp_grid

def crossover(grid1, grid2, N):
    """
    picks a random row - R.
    Append to grid3 all rows from grid1 until R and all rows from grid2 from R until the end
    """
    row = random.randint(0, N)
    grid3 = grid1[:row]
    grid3 += grid2[row:]
    return grid3

class SolutionFactory:
    def __init__(self):
        pass

    def get_solution(self):
        return {}

class StrategySolver:
    def __init__(self):
        pass

if __name__ == '__main__':
    game_dict = parse_input(io=None)
    n = game_dict['N']
    grid = make_grid2(n)
    # g = [[0 if (i % 2 == 0 and j % 2 == 0) else ' ' for j in range(n + n -1)] for i in range(n + n - 1)]
    # g = [[j for j in range(n)] for i in range(n)]
    constraint = [((1, 1), (1, 2))]
    # print_grid(grid, constraint)
    new_grid = mutate(grid, n)
    print_grid(grid, constraint)
    print_grid(new_grid, constraint)
    new_new_grid = crossover(grid, new_grid, n)
    print_grid(new_new_grid, constraint)

