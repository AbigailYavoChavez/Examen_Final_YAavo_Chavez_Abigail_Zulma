import numpy as np
import random
import time

class PuzzleEnvironment:
    def __init__(self):
        self.rows = 4
        self.cols = 5
        self.state = np.zeros((self.rows, self.cols))  # Estado inicial
        self.goal_state = np.ones((self.rows, self.cols))  # Estado objetivo
        self.actions = ['arriba', 'abajo', 'izquierda', 'derecha']

        self.agent_position = [0, 0]  # Inicializar la posición del agente
        
    def reset(self):
        self.state = np.zeros((self.rows, self.cols))
        self.agent_position = [0, 0]  # Reiniciar la posición del agente
        return self.state
    
    def step(self, action):
        reward = -1
        done = False
        row, col = self.agent_position
        
        if action == 'up' and row > 0:
            self.agent_position[0] -= 1
        elif action == 'down' and row < self.rows - 1:
            self.agent_position[0] += 1
        elif action == 'left' and col > 0:
            self.agent_position[1] -= 1
        elif action == 'right' and col < self.cols - 1:
            self.agent_position[1] += 1

        # Actualizar el estado con la nueva posición del agente
        self.state = np.zeros((self.rows, self.cols))
        self.state[tuple(self.agent_position)] = 1

        # Verificar si el agente ha alcanzado el estado objetivo
        if np.array_equal(self.state, self.goal_state):
            reward = 10
            done = True

        # Verificar si se ha excedido el límite de pasos
        if not done and action != 'reset':
            if self.step_count >= 1000:
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
    
    def get_q(self, state, action): # Retorna el valor Q para una pareja estado-acción. Si no existe, retorna 0.
        return self.q_table.get((tuple(state.flatten()), action), 0.0)
    
    def update_q(self, state, action, reward, next_state): #Actualiza el valor Q usando la fórmula de Q-learning.
        best_next_action = max(self.env.actions, key=lambda a: self.get_q(next_state, a))
        td_target = reward + self.gamma * self.get_q(next_state, best_next_action)
        td_error = td_target - self.get_q(state, action)
        self.q_table[(tuple(state.flatten()), action)] = self.get_q(state, action) + self.alpha * td_error
    
    def choose_action(self, state): #Selecciona una acción basada en la política epsilon-greedy.
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return max(self.env.actions, key=lambda action: self.get_q(state, action))
    
    def train(self, episodes):#Entrena al agente a lo largo de varios episodios. Durante cada episodio, el agente realiza acciones, actualiza su tabla Q y almacena los resultados. También imprime mensajes de depuración y el progreso del entrenamiento.
        start_time = time.time()  # Capturar el tiempo de inicio aquí
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            self.env.step_count = 0  # Inicializar el contador de pasos para este episodio
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q(state, action, reward, next_state)
                state = next_state
                self.env.step_count += 1
                # Almacenar los pasos y recompensas por episodio
                if done:
                    self.episode_steps.append(self.env.step_count)
                    self.episode_rewards.append(reward)
                
                # Mensajes de depuración adicionales
                if self.env.step_count % 10 == 0:
                    print(f"Episodio {episode + 1}, Paso {self.env.step_count}")
                    self.env.render()
                    print(f"Acción: {action}, Recompensa: {reward}, Completado: {done}")
                    print("---")
                # Condición para terminar el episodio si se excede un número máximo de pasos
                if self.env.step_count >= 1000:
                    print(f"Episodio {episode + 1} terminado debido al límite de pasos")
                    done = True  # Marcar done como True para salir del bucle
            
            if (episode + 1) % 100 == 0:
                print(f"Episodio {episode + 1} completado, pasos realizados: {self.env.step_count}")
                print(f"Tamaño de la tabla Q: {len(self.q_table)}")
        
        end_time = time.time()
        print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos")  # Imprimir el tiempo total de entrenamiento
        
        # Mostrar resultados al final del entrenamiento
        self.print_training_results()

    def print_training_results(self): # Muestra los resultados del entrenamiento, incluyendo los pasos y recompensas por episodio.
        print("Resultados del entrenamiento:")
        print("Pasos por episodio:", self.episode_steps)
        print("Recompensas por episodio:", self.episode_rewards)

if __name__ == "__main__": #Ejecución del Código Principal - Esta estructura completa proporciona un ciclo de entrenamiento y prueba para un agente que aprende a resolver un rompecabezas utilizando Q-learning.
    print("Iniciando entrenamiento...")
    env = PuzzleEnvironment()
    agent = QLearningAgent(env)
    
    # Entrenar al agente
    agent.train(10)
    
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

    
