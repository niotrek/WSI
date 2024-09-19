import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def train_agent(alpha, gamma, epsilon, tau, epsilon_min, epsilon_decay, tau_min, tau_decay, num_episodes=2000, seed=42, boltzmann=False):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    env.reset(seed=seed)
    np.random.seed(seed)

    n_actions = env.action_space.n
    n_states = env.observation_space.n

    Q = np.zeros((n_states, n_actions))
    rewards = []

    def choose_action(state, epsilon, tau):
        if boltzmann:
            q_values = Q[state]
            exp_q = np.exp(q_values / tau)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(np.arange(n_actions), p=probs)
        else:
            if np.random.rand() < epsilon:
                return np.random.choice(n_actions)
            else:
                max_q = np.max(Q[state])
                best_actions = np.where(Q[state] == max_q)[0]
                return np.random.choice(best_actions)
                

    for episode in range(num_episodes):
        state, info = env.reset(seed=seed)
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state, epsilon, tau)
            next_state, reward, terminated, truncated, info = env.step(action)
            max_q_value = np.max(Q[next_state])
            best_actions = np.where(Q[next_state] == max_q_value)[0]
            best_next_action = np.random.choice(best_actions)
            # best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state
            total_reward += reward

            if terminated or truncated:
                done = True
            
        if boltzmann == False:
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        else:
            if tau > tau_min:
                tau *= tau_decay

        rewards.append(total_reward)

    env.close()

    return Q, rewards

def evaluate(param_value, param_name, fixed_params, boltzmann=False):
    rewards_per_seed = []
    for seed in range(5):
        params = fixed_params.copy()
        params[param_name] = param_value
        _, rewards = train_agent(**params, seed=seed, boltzmann=boltzmann)
        rewards_per_seed.append(rewards)

    avg_rewards_per_episode = np.mean(rewards_per_seed, axis=0)
    std_rewards_per_episode = np.std(rewards_per_seed, axis=0)
    print(f"Finished evaluation for {param_name} = {param_value}")
    print(f"Average reward: {avg_rewards_per_episode}")
    print(f"Standard deviation: {std_rewards_per_episode}")
    return avg_rewards_per_episode, std_rewards_per_episode


def plot_results(avg_rewards_per_episode, std_rewards_per_episode, param_name, param_value, filepath):
    # episodes = np.arange(len(avg_rewards_per_episode))
    episodes = np.arange(0, len(avg_rewards_per_episode), 10)
    avg_rewards_per_episode = avg_rewards_per_episode[::10]
    std_rewards_per_episode = std_rewards_per_episode[::10]
    plt.plot(episodes, avg_rewards_per_episode, label='Average Reward')
    # plt.errorbar(episodes, avg_rewards_per_episode, yerr=std_rewards_per_episode, fmt='none', ecolor='black', capsize=1, linewidth=0.2, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Average Reward per Episode\n{param_name} = {param_value}")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def calculate_statistics(rewards, param_name, param_value):
    last_10_rewards = rewards[-10:]
    avg_last_10 = np.mean(last_10_rewards)
    std_dev_last_10 = np.std(last_10_rewards)
    successful_episodes = np.sum(np.array(last_10_rewards) > 0.2)
    return {
        'param_name': param_name,
        'param_value': param_value,
        'avg_last_10': avg_last_10,
        'std_dev_last_10': std_dev_last_10,
        'successful_episodes': successful_episodes
    }


alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
gamma_values = [0.5, 0.7, 0.9, 0.95, 0.99]
epsilon_values = [0.2, 0.4, 0.6, 0.8, 1.0]
tau_values = [0.1, 0.5, 1.0, 2.0, 5.0]

fixed_params = {
    "alpha": 0.5,
    "gamma": 0.99,
    "epsilon": 0.6,
    "epsilon_min" : 0.01,  
    "epsilon_decay" : 0.995,  
    "tau": 1.0,
    "tau_min" : 0.1,  
    "tau_decay" : 0.995,
    "num_episodes": 5000
}

param_value_dict = {
    "alpha": alpha_values,
    "gamma": gamma_values,
    "epsilon": epsilon_values,
    "tau": tau_values
}

output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)
statistics = []

for param_name in param_value_dict:
    boltzmann = (param_name == "tau")
    param_values = param_value_dict[param_name]
    for param_value in param_values:
        results, std_devs = evaluate(param_value, param_name, fixed_params, boltzmann=boltzmann)
        stats = calculate_statistics(results, param_name, param_value)
        statistics.append(stats)
        #filename = f"{param_name}_{param_value}.png"
        #filepath = os.path.join(output_folder, filename)
        #plot_results(results, std_devs, param_name, param_value, filepath)
        
df = pd.DataFrame(statistics)
grouped = df.groupby('param_name')

for param_name, group in grouped:
    print(f"Param Name: {param_name}")
    print(group)
    with open(f'{param_name}.tex', 'w') as tf:
         tf.write(group.to_latex())