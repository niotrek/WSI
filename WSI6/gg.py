import gym
import numpy as np
import random

# Parametry Q-learningu
alpha = 0.1  # Współczynnik uczenia
gamma = 0.99  # Współczynnik dyskontowania
epsilon = 1.0  # Parametr eksploracji
epsilon_min = 0.01  # Minimalna wartość epsilon
epsilon_decay = 0.995  # Tempo zmniejszania epsilon
num_episodes = 2000  # Liczba epizodów
tau = 1.0  # Początkowa temperatura dla rozkładu Boltzmanna
tau_min = 0.1  # Minimalna wartość tau
tau_decay = 0.995  # Tempo zmniejszania tau

# Inicjalizacja środowiska
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
n_actions = env.action_space.n
n_states = env.observation_space.n

# Inicjalizacja tablicy Q
Q = np.zeros((n_states, n_actions))

# Funkcja wybierania działania przy użyciu epsilon-greedy
def choose_action_greedy(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Wybierz losowe działanie
    else:
        return np.argmax(Q[state])  # Wybierz działanie o najwyższej wartości Q
    
# Funkcja wybierania działania przy użyciu rozkładu Boltzmanna
def choose_action_boltzmann(state, tau):
    q_values = Q[state]
    exp_q = np.exp(q_values / tau)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(np.arange(n_actions), p=probs)

# Główna pętla Q-learningu
for episode in range(num_episodes):
    state, info = env.reset(seed=42)
    # print("State: ", state)
    done = False
    while not done:
        action = choose_action_boltzmann(state, tau)
        next_state, reward, terminated, truncated, info = env.step(action)
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action]
        td_error = td_target - Q[int(state), int(action)]  # Upewnij się, że state i action są intami
        Q[int(state), int(action)] += alpha * td_error
        state = next_state

        if terminated or truncated:
            break

    # Redukcja tau
    if tau > tau_min:
        tau *= tau_decay

print("Q-Table:")
print(Q)
env.close()

# Generacja środowiska
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="human")

# Testowanie wytrenowanego agenta
state, info = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(Q[int(state)])
    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        done = True  
env.close()
