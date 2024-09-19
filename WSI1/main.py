import numpy as np  
import matplotlib.pyplot as plt

def rastrigin(x1, x2):
    return 20 + x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))

def grad_rastrigin(x1, x2):
    dfdx1 = 2*x1 + 10*np.sin(2*np.pi*x1)*2*np.pi
    dfdx2 = 2*x2 + 10*np.sin(2*np.pi*x2)*2*np.pi
    return (dfdx1, dfdx2)

def griewank(x1, x2):
    return 1 + x1**2/4000 + x2**2/4000 - np.cos(x1)*np.cos(x2/np.sqrt(2))

def grad_griewank(x1, x2):
    dfdx1 = x1/2000 + np.sin(x1)*np.cos(x2/np.sqrt(2))
    dfdx2 = x2/2000 + np.sin(x2/np.sqrt(2))*np.cos(x1)/np.sqrt(2) 
    return (dfdx1, dfdx2)

def sum_of_squares(x1, x2):
    return x1**2 + x2**2

def grad_sum_of_squares(x1, x2):
    return (2*x1, 2*x2)

def gradient_descent(fun, gradient, learning_rate, current_pos):
    values = []
    for _ in range(1000):
        dx, dy = gradient(current_pos[0], current_pos[1])
        new_x = current_pos[0] - learning_rate * dx
        new_y = current_pos[1] - learning_rate * dy
        new_z = fun(new_x, new_y)
        current_pos = (new_x, new_y, new_z)
        values.append(current_pos[2])

        if np.linalg.norm((dx,dy)) < 1e-5:
            break
    print(f"Final value: {current_pos[2]}")
    print(f"Iterations: {len(values)}")
    return values

def visualize_function(fun, gradient, x, learning_rate, current_pos):
    X, Y = np.meshgrid(x, x)
    Z = fun(X, Y)
    ax = plt.subplot(projection='3d', computed_zorder=False)
    for _ in range(500):
        dx, dy = gradient(current_pos[0], current_pos[1])
        new_x = current_pos[0] - learning_rate * dx
        new_y = current_pos[1] - learning_rate * dy
        new_z = fun(new_x, new_y)
        current_pos = (new_x, new_y, new_z)
        if np.linalg.norm((dx,dy)) < 1e-3:
            break
        ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
        ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='red',zorder=1)
        plt.pause(0.001)
        ax.clear()
    #plt.show()
    plt.close()
    print(f"Final position: {current_pos}")

def main():
    x = np.arange(-10, 10, 0.05)
    visualize_function(sum_of_squares, grad_sum_of_squares, x, 0.1, (10, -10))
    
    x = np.arange(-5.12, 5.12, 0.05)
    visualize_function(rastrigin, grad_rastrigin, x, 0.001, (0.5, 1.5))
   
    x = np.arange(-5, 5, 0.05)
    visualize_function(griewank, grad_griewank, x, 0.1, (1, 2))      
    
    plt.figure()
    learning_rates = [0.01, 0.001, 0.0001]
    for learning_rate in learning_rates:
        values = gradient_descent(rastrigin, grad_rastrigin, learning_rate, (0.5, 0.5))
        plt.plot(values, label=f'Learning rate: {learning_rate}')
    plt.xlabel('Iterations')
    plt.ylabel('Objective function value')
    plt.title('Rastrigin function')
    plt.legend()
    plt.show()
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    plt.figure()
    for learning_rate in learning_rates:
        values = gradient_descent(griewank, grad_griewank, learning_rate, (1, 1))
        plt.plot(values, label=f'Learning rate: {learning_rate}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Objective function value')
    plt.title('Griewank function')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()