import numpy as np
import random
import time

class PuzzleEnvironment:
    def __init__(self):
        self.rows = 4
        self.cols = 5
        self.state = np.random.permutation(np.arange(1, 21)).reshape(self.rows, self.cols)  # Estado inicial desordenado
        self.goal_state = np.arange(1, 21).reshape(self.rows, self.cols)  # Estado objetivo ordenado del 1 al 20
        self.actions = ['arriba', 'abajo', 'izquierda', 'derecha']
        self.agent_position = np.argwhere(self.state == 1)[0]  # Posición inicial del agente
        
    def reset(self):
        self.state = np.random.permutation(np.arange(1, 21)).reshape(self.rows, self.cols)
        self.agent_position = np.argwhere(self.state == 1)[0]
        return self.state
    
    def step(self, action):
        reward = -1  # Recompensa por defecto por cada acción
        done = False
        row, col = self.agent_position
        
        # Movimiento del agente
        if action == 'up' and row > 0:
            self.state[row, col], self.state[row - 1, col] = self.state[row - 1, col], self.state[row, col]
            self.agent_position = [row - 1, col]
        elif action == 'down' and row < self.rows - 1:
            self.state[row, col], self.state[row + 1, col] = self.state[row + 1, col], self.state[row, col]
            self.agent_position = [row + 1, col]
        elif action == 'left' and col > 0:
            self.state[row, col], self.state[row, col - 1] = self.state[row, col - 1], self.state[row, col]
            self.agent_position = [row, col - 1]
        elif action == 'right' and col < self.cols - 1:
            self.state[row, col], self.state[row, col + 1] = self.state[row, col + 1], self.state[row, col]
            self.agent_position = [row, col + 1]

        # Verificar si ha alcanzado el estado objetivo
        if np.array_equal(self.state, self.goal_state):
            reward = 10  # Recompensa positiva por ordenar correctamente
            done = True

        return self.state, reward, done
    
    def render(self):
        print("Estado actual del rompecabezas:")
        print(self.state)


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.env = env
        self.q_table = {}
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de elegir una acción aleatoria
        self.episode_steps = []  # Para almacenar los pasos por episodio
        self.episode_rewards = []  # Para almacenar las recompensas por episodio
    
    def get_q(self, state, action):
        return self.q_table.get((tuple(state.flatten()), action), 0.0)
    
    def update_q(self, state, action, reward, next_state):
        best_next_action = max(self.env.actions, key=lambda a: self.get_q(next_state, a))
        td_target = reward + self.gamma * self.get_q(next_state, best_next_action)
        td_error = td_target - self.get_q(state, action)
        self.q_table[(tuple(state.flatten()), action)] = self.get_q(state, action) + self.alpha * td_error
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return max(self.env.actions, key=lambda action: self.get_q(state, action))
    
    def train(self, episodes, max_steps_per_episode=1000):
        start_time = time.time()
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            step_count = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q(state, action, reward, next_state)
                state = next_state
                step_count += 1
                
                # Almacenar los pasos y recompensas por episodio
                if done:
                    self.episode_steps.append(step_count)
                    self.episode_rewards.append(reward)
                
                # Mensajes de depuración adicionales
                if step_count % 10 == 0:
                    print(f"Episodio {episode}, Paso {step_count}")
                    self.env.render()
                    print(f"Acción: {action}, Recompensa: {reward}, Completado: {done}")
                    print("---")
                
                # Condición para terminar el episodio si se excede un número máximo de pasos
                if step_count >= max_steps_per_episode:
                    print(f"Episodio {episode} terminado debido al límite de pasos")
                    break
            
            # Mensaje al completar cada episodio
            print(f"Episodio {episode} completado, pasos realizados: {step_count}")
            print(f"Tamaño de la tabla Q: {len(self.q_table)}")
        
        end_time = time.time()
        print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos")
        
        self.print_training_results()

    def print_training_results(self):
        print("Resultados del entrenamiento:")
        print("Pasos por episodio:", self.episode_steps)
        print("Recompensas por episodio:", self.episode_rewards)

if __name__ == "__main__":
    print("Iniciando entrenamiento...")
    env = PuzzleEnvironment()
    agent = QLearningAgent(env)
    
    # Entrenar al agente
    agent.train(episodes=10)
    
    print("Entrenamiento completado, comenzando prueba...")
    
    # Probar al agente
    state = env.reset()
    env.render()  # Mostrar el entorno inicial
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        env.render()  # Mostrar el entorno después de cada acción
        print(f"Acción: {action}, Recompensa: {reward}")

    print("Prueba completada")
