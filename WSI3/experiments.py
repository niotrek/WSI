from ai import position, minimax, alpha_beta
import time
import statistics
from matplotlib import pyplot as plt


def main():
    board = ['X', ' ', ' ', ' ', ' ', ' ', ' ', 'O', ' ']
    pos = position(board)
    pos.print_board()
    minimax_time = []
    alpha_beta_time = []
    for _ in range(10):
        start_time = time.time()
        minimax(pos,9, False)
        end_time = time.time()
        execution_time = end_time - start_time
        minimax_time.append(execution_time)
        print("Minimax execution time:", execution_time)
        start_time = time.time()
        alpha_beta(pos,9, float('-inf'), float('inf'), False, -1, [], 0)
        end_time = time.time()
        execution_time = end_time - start_time
        alpha_beta_time.append(execution_time)
        print("Alpha-beta execution time:", execution_time)

    average_time_minimax = statistics.mean(minimax_time)
    std_dev_time_minimax = statistics.stdev(minimax_time)
    print("Minimax average execution time:", average_time_minimax)
    print("Minimax standard deviation:", std_dev_time_minimax)
    avarage_time_alpha_beta = statistics.mean(alpha_beta_time)
    std_dev_time_alpha_beta = statistics.stdev(alpha_beta_time)
    print("Alpha-beta average execution time:", avarage_time_alpha_beta)
    print("Alpha-beta standard deviation:", std_dev_time_alpha_beta)

    game_states = [['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     #['X', ' ', ' ', ' ', 'O', ' ', ' ', ' ', ' '],
                     ['X', 'X', ' ', ' ', 'O', ' ', ' ', ' ', ' '],
                     #['X', 'X', 'O', ' ', 'O', ' ', ' ', ' ', ' '],
                     ['X', 'X', 'O', ' ', 'O', ' ', 'X', ' ', ' '],
                     #['X', 'X', 'O', 'O', 'O', ' ', 'X', ' ', ' '],
                     ['X', 'X', 'O', 'O', 'O', 'X', 'X', ' ', ' ']]
                    # ['X', 'X', 'O', 'O', 'O', 'X', 'X', 'O', ' '],
                     #['X', 'X', 'O', 'O', 'O', 'X', 'X', 'O', 'X']]
    
    times_minimax = []
    times_alpha_beta = []
    for state in game_states:
        start_time = time.time()
        minimax(position(state), 9, False)
        end_time = time.time()
        execution_time = end_time - start_time
        times_minimax.append(execution_time)

        start_time = time.time()
        alpha_beta(position(state), 9, float('-inf'), float('inf'), False, -1, [], 0)
        end_time = time.time()
        execution_time = end_time - start_time
        times_alpha_beta.append(execution_time)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9, 2), times_minimax, label='Minimax')
    plt.plot(range(1, 9, 2), times_alpha_beta, label='Alpha-beta')
    plt.xlabel('Game State')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.show()
    plt.savefig('execution_time.png')
    
    boards = [['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
              [' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
              [' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '], 
              [' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' '], 
              [' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' '], 
              [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' '], 
              [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' '], 
              [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' '], 
              [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], 
              ['X', 'O', ' ', ' ', 'X', ' ', ' ', ' ', ' '],
              ['X', 'O', 'O', 'X', ' ', ' ', ' ', 'X', ' '],
              [' ', ' ', ' ', ' ', ' ', ' ', 'X', 'O', 'X'] ]
    
    node_count_minimax_table = []
    node_count_alpha_beta_table = []
    depth_avarage = []
    depths = []
    for element in boards:
        pos = position(element)
        pos.print_board()
        _, _, node_count_minimax = minimax(pos, 9, False)
        _, _, depths, node_count_alpha_beta = alpha_beta(pos, 9, float('-inf'), float('inf'), False, -1, depths, 0)
        node_count_minimax_table.append(node_count_minimax)
        node_count_alpha_beta_table.append(node_count_alpha_beta)
        print(depths)
        depth_avarage.append(statistics.mean(depths))
        depths.clear()
    
    avarage_depth = statistics.mean(depth_avarage[:9])
    standard_deviation_depth = statistics.stdev(depth_avarage[:9])
    avarage_node_count_minimax = statistics.mean(node_count_minimax_table[:9])
    avarage_node_count_alpha_beta = statistics.mean(node_count_alpha_beta_table[:9])
    standard_deviation_node_count_minimax = statistics.stdev(node_count_minimax_table[:9])
    standard_deviation_node_count_alpha_beta = statistics.stdev(node_count_alpha_beta_table[:9])



    with open('results.txt', 'w') as f:
        f.write('Minimax average execution time: ' + str(average_time_minimax) + '\n')
        f.write('Minimax standard deviation: ' + str(std_dev_time_minimax) + '\n')
        for element in minimax_time:
            f.write(str(element) + '\n')
        f.write('Alpha-beta average execution time: ' + str(avarage_time_alpha_beta) + '\n')
        f.write('Alpha-beta standard deviation: ' + str(std_dev_time_alpha_beta) + '\n')
        for element in alpha_beta_time:
            f.write(str(element) + '\n')
        f.write('Minimax node count table: ' + str(node_count_minimax_table) + '\n')
        f.write('Alpha-beta node count table: ' + str(node_count_alpha_beta_table) + '\n')
        f.write('Average depth: ' + str(depth_avarage) + '\n')
        f.write('Average depth: ' + str(avarage_depth) + '\n')
        f.write('Standard deviation depth: ' + str(standard_deviation_depth) + '\n')
        f.write('Average node count minimax: ' + str(avarage_node_count_minimax) + '\n')
        f.write('Average node count alpha-beta: ' + str(avarage_node_count_alpha_beta) + '\n')
        f.write('Standard deviation node count minimax: ' + str(standard_deviation_node_count_minimax) + '\n')
        f.write('Standard deviation node count alpha-beta: ' + str(standard_deviation_node_count_alpha_beta) + '\n')


    


if __name__ == '__main__':
    main()