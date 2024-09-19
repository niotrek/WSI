import numpy as np
import matplotlib.pyplot as plt
import random

def rastrigin(x1, x2):
    return 20 + x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))

def griewank(x1, x2):
    return 1 + x1**2/4000 + x2**2/4000 - np.cos(x1)*np.cos(x2/np.sqrt(2))

def drop_wave(x1, x2):
    return -(1 + np.cos(12*np.sqrt(x1**2 + x2**2))) / (0.5*(x1**2 + x2**2) + 2)

def fitness_function(function, x1, x2):
    return 1/(function(x1, x2))

def visualize_best(function, trajectory, range):
    min_x, min_y, min_z = trajectory[-1]
    MIN_X = range
    MAX_X = range
    PLOT_STEP = 100

    x1 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    x2 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    X1, X2 = np.meshgrid(x1, x2)
    Z = function(X1, X2)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
    plt.colorbar(label='Objective Function Value')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Visualization')
    plt.scatter(min_x, min_y, color='yellow', label='Minimum found by evolution algorithm.', zorder = 3)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Evolution algorithm steps', alpha=0.5)
    plt.legend()
    plt.show()

def visualize_two_populations(populations, function, ranges):
    MIN_X = ranges
    MAX_X = ranges
    PLOT_STEP = 100

    x1 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    x2 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    X1, X2 = np.meshgrid(x1, x2)
    Z = function(X1, X2)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    title = ['first', 'last']

    for i in range(2):
        axs[i].pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
        axs[i].set_xlabel('x1')
        axs[i].set_ylabel('x2')
        axs[i].set_title(f'Objective Function Visualization for {title[i]} population')
        for individual in populations[i]:
            axs[i].scatter(individual[0], individual[1], color='red', zorder = 3)

    plt.show()

def visualize_populations(populations, function, range_value):
    MIN_X = range_value
    MAX_X = range_value
    PLOT_STEP = 100

    x1 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    x2 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    X1, X2 = np.meshgrid(x1, x2)
    Z = function(X1, X2)

    fig, ax = plt.subplots(figsize=(8, 6))
    pc = ax.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
    sc = ax.scatter([], [], color='red', zorder = 3)  

    for i, population in enumerate(populations):
        sc.set_offsets(population)  
        ax.set_title(f'Objective Function Visualization for population {i+1}')
        plt.draw()
        plt.pause(0.2)  
        if i == 49:
            break

    plt.show()

def mutation(x, p_mutation, sigma, function):
    x_result = []
    for xi in x:
        if np.random.rand() < p_mutation:
            x_mutated = []
            for xii in xi[:-1]:
                x_mutated.append(xii + np.random.normal(0, sigma))
            x_mutated.append(fitness_function(function, x_mutated[0], x_mutated[1]))
            x_result.append(x_mutated)
    return x_result
    
def crossover(x, p_crossover, function):
    x_crossover = []
    for x1 in x:
        if np.random.rand() < p_crossover:
            x_child = []
            x2 = random.choice(x)
            for xi1, xi2 in zip(x1[:-1], x2[:-1]):
                w = np.random.rand()
                x_child.append(w*xi1 + (1-w)*xi2)
            x_child.append(fitness_function(function, x_child[0], x_child[1]))
            x_crossover.append(x_child)
    return x_crossover
           
def reproduction(population, limit):
    winners = []
    for _ in range(int(limit)):
         individual1 = random.choice(population)
         individual2 = random.choice(population)
         if individual1[2] > individual2[2]:
             winners.append(individual1)
         else:
             winners.append(individual2)
    return winners

def succession(population, k, mutants):
    population.sort(key=lambda x: x[2], reverse=True)
    population = population[:k]
    population.extend(mutants)
    population.sort(key=lambda x: x[2], reverse=True)
    population = population[:-k]
    return population
    
def genetic_algorithm(population_size, p_mutation, p_crossover, sigma, function, limit, k, iterations, ranges):
    population = []
    trajectory = []
    for _ in range(int(population_size)):
        x1 = np.random.uniform(-ranges, ranges)
        x2 = np.random.uniform(-ranges, ranges)
        population.append([x1, x2, fitness_function(function, x1, x2)])
    population.sort(key=lambda x: x[2], reverse=True)
    best = population[0]
    trajectory.append(best)
    first_and_last = []
    first_and_last.append(population)
    all_populations = []
    all_populations.append(population)
    for _ in range(int(iterations)):
        winners = reproduction(population, limit)
        mutants = mutation(winners, p_mutation, sigma, function)
        crossovers = crossover(winners, p_crossover, function)
        new_population = winners + mutants + crossovers
        population = succession(population, k, new_population)
        all_populations.append(population)
        if population[0][2] > best[2]:
            best = population[0]
            trajectory.append(best)
    first_and_last.append(population)
    return best, np.array(trajectory), first_and_last, all_populations

def main():
    # solution_rastrigin = genetic_algorithm(200, 0.2, 0.8, 0.2, rastrigin, 200, 30, 10000, 5.12)
    # solution_griewank = genetic_algorithm(500, 0.5, 0.8, 0.8, griewank, 500, 10, 10000, 50)
    # solution_drop_wave = genetic_algorithm(200, 0.2, 0.8, 0.2, drop_wave, 200, 30, 10000, 5.12)
    # print(f"Rastrigin: x1 {solution_rastrigin[0][0]} x2 {solution_rastrigin[0][1]} f(x1, x2) {rastrigin(solution_rastrigin[0][0], solution_rastrigin[0][1])}")
    # print(f"Griewank: x1 {solution_griewank[0][0]} x2 {solution_griewank[0][1]} f(x1, x2) {griewank(solution_griewank[0][0], solution_griewank[0][1])}")
    # print(f"Drop wave: x1 {solution_drop_wave[0][0]} x2 {solution_drop_wave[0][1]} f(x1, x2) {drop_wave(solution_drop_wave[0][0], solution_drop_wave[0][1])}")
    # visualize_best(rastrigin, solution_rastrigin[1], 5.12)
    
    # visualize_best(griewank, solution_griewank[1], 50)
    # visualize_best(drop_wave, solution_drop_wave[1], 5.12)
    # visualize_populations(solution_rastrigin[3], rastrigin, 5.12)
    # visualize_populations(solution_griewank[3], griewank, 50)
    # visualize_populations(solution_drop_wave[3], drop_wave, 5.12)
    # visualize_two_populations(solution_rastrigin[2], rastrigin, 5.12)
    # visualize_two_populations(solution_griewank[2], griewank, 50)
    # visualize_two_populations(solution_drop_wave[2], drop_wave, 5.12)

    functions = [
    {
        "func": rastrigin,
        "pop_size": 200,
        "mutation_prob": 0.2,
        "crossover_prob": 0.8,
        "sigma": 0.2,
        "winners": 100,
        "k": 30,
        "iterations": 200,
        "bound": 5.12
    },
    {
        "func": griewank,
        "pop_size": 400,
        "mutation_prob": 0.5,
        "crossover_prob": 0.8,
        "sigma": 0.8,
        "winners": 400,
        "k": 10,
        "iterations": 1000,
        "bound": 50
    },
    {
        "func": drop_wave,
        "pop_size": 100,
        "mutation_prob": 0.2,
        "crossover_prob": 0.8,
        "sigma": 0.2,
        "winners": 100,
        "k": 30,
        "iterations": 100,
        "bound": 5.12
    }
    ]

    for params in functions:
        solution = genetic_algorithm(
        params["pop_size"], 
        params["mutation_prob"], 
        params["crossover_prob"], 
        params["sigma"], 
        params["func"], 
        params["winners"], 
        params["k"], 
        params["iterations"], 
        params["bound"]
        )
        print(f"{params['func'].__name__}: x1 {solution[0][0]} x2 {solution[0][1]} f(x1, x2) {params['func'](solution[0][0], solution[0][1])}")
        visualize_best(params['func'], solution[1], params['bound'])
        visualize_populations(solution[3], params['func'], params['bound'])
        visualize_two_populations(solution[2], params['func'], params['bound'])

        iterations = [10, 20, 50, 100, 200]
        final_values = []

        for i in iterations:
            solution = genetic_algorithm(
                params["pop_size"], 
                params["mutation_prob"], 
                params["crossover_prob"], 
                params["sigma"], 
                params["func"], 
                params["winners"], 
                params["k"], 
                i, 
                params["bound"]
            )
            
            final_values.append(params['func'](solution[0][0], solution[0][1]))
            # if i == 10 or i == 50 or  i == 200:
            #     visualize_two_populations(solution[2], params['func'], params['bound'])
            #     plt.savefig(f'iterations_{i}_{params['func'].__name__}.png')
            
            print("Final value for iteration", i, "is", params['func'](solution[0][0], solution[0][1]), "for x1", solution[0][0], "and x2", solution[0][1])

        # plt.figure(figsize=(8, 6))
        # plt.plot(iterations, final_values, marker='o')
        # plt.xlabel('Iterations')
        # plt.ylabel('Final Objective Function Value')
        # plt.title('Final Objective Function Value vs Iterations')
        # plt.grid(True)
        # plt.show()
        # plt.savefig(f'iterations_{params['func'].__name__}.png')

        mutation_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        final_values = []

        for i in mutation_probs:
            solution = genetic_algorithm(
                params["pop_size"], 
                i, 
                params["crossover_prob"], 
                params["sigma"], 
                params["func"], 
                params["winners"], 
                params["k"], 
                params["iterations"], 
                params["bound"]
            )
            # if i == 0.1 or i == 0.5 or i == 0.9:
            #     visualize_two_populations(solution[2], params['func'], params['bound'])
            #     plt.savefig(f'p_mutation_{i}_{params['func'].__name__}.png')
            final_values.append(params['func'](solution[0][0], solution[0][1]))
            print("Final value for mutation probability", i, "is", params['func'](solution[0][0], solution[0][1]), "for x1", solution[0][0], "and x2", solution[0][1])
        
        # plt.figure(figsize=(8, 6))
        # plt.plot(mutation_probs, final_values, marker='o')
        # plt.xlabel('Mutation Probability')
        # plt.ylabel('Final Objective Function Value')
        # plt.title('Final Objective Function Value vs Mutation Probability')
        # plt.grid(True)
        # plt.show()
        # plt.savefig(f'p_mutation_{params['func'].__name__}.png')

        crossover_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        final_values = []

        for i in crossover_probs:
            solution = genetic_algorithm(
                params["pop_size"], 
                params["mutation_prob"], 
                i, 
                params["sigma"], 
                params["func"], 
                params["winners"], 
                params["k"], 
                params["iterations"], 
                params["bound"]
            )

            # if i == 0.1 or i == 0.5 or i == 0.9:
            #     visualize_two_populations(solution[2], params['func'], params['bound'])
            #     plt.savefig(f'p_crossover_{i}_{params['func'].__name__}.png')
            final_values.append(params['func'](solution[0][0], solution[0][1]))
            print("Final value for crossover probability", i, "is", params['func'](solution[0][0], solution[0][1]), "for x1", solution[0][0], "and x2", solution[0][1])

        # plt.figure(figsize=(8, 6))
        # plt.plot(crossover_probs, final_values, marker='o')
        # plt.xlabel('Crossover Probability')
        # plt.ylabel('Final Objective Function Value')
        # plt.title('Final Objective Function Value vs Crossover Probability')
        # plt.grid(True)
        # plt.show()
        # plt.savefig(f'p_crossover_{params['func'].__name__}.png')

        population_sizes = [2, 10, 50, 100, 200, 400]
        final_values = []

        for i in population_sizes:
            solution = genetic_algorithm(
                i, 
                params["mutation_prob"], 
                params["crossover_prob"], 
                params["sigma"], 
                params["func"], 
                params["winners"], 
                params["k"], 
                params["iterations"], 
                params["bound"]
            )
            # if i == 10 or i == 50 or i == 200:
            #     visualize_two_populations(solution[2], params['func'], params['bound'])
            #     plt.savefig(f'pop_size_{i}_{params['func'].__name__}.png')
            
            final_values.append(params['func'](solution[0][0], solution[0][1]))
            print("Final value for population size", i, "is", params['func'](solution[0][0], solution[0][1]), "for x1", solution[0][0], "and x2", solution[0][1])
        
        # plt.figure(figsize=(8, 6))
        # plt.plot(population_sizes, final_values, marker='o')
        # plt.xlabel('Population Size')
        # plt.ylabel('Final Objective Function Value')
        # plt.title('Final Objective Function Value vs Population Size')
        # plt.grid(True)
        # plt.show()
        # plt.savefig(f'pop_size_{params['func'].__name__}.png')





        
    
 

if __name__ == "__main__":
    main()