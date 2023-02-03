# Author: Marco Kemmerling
# Email: marco.kemmerling@ima.rwth-aachen.de
# The code is written for minimization problems. You will have to make some changes for maximization problems.
# -----------------------------------------------------------

import random
import math


class SimulatedAnnealingOptimizer():
    """
    Metaheuristic. Used for post-optimization of RL results.
    Note that the time per stop is specified in Order.py
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def optimize(self, current_solution, temp=1000, cooling_rate=0.005):
        """
        Further optimizes the initial solution given by 'current_solution'
        temp: initial temperature
        cooling_rate: determines how much cooling rate is decreased each step.
        High temperatures increase the probability that the algorithm will accept a solution worse than the current one, which is helpful
        in escaping local optima.
        """

        best_solution = current_solution[:]
        best_energy = self.energy(current_solution)

        if self.verbose:
            print('Initial Energy:', best_energy)

        while temp > 1.0 and best_energy > 0:
            new_solution = self.step(current_solution[:])
            current_energy = self.energy(current_solution)

            new_energy = self.energy(new_solution)

            if self.acceptance_prob(current_energy, new_energy, temp) > random.random():
                current_solution = new_solution[:]
                current_energy = new_energy

            if current_energy < best_energy:
                best_solution = current_solution[:]
                best_energy = current_energy

            temp *= 1 - cooling_rate

        if self.verbose:
            print('Best Energy', best_energy)

        return best_solution, best_energy

    def acceptance_prob(self, energy, new_energy, temp):
        if new_energy < energy:
            return 1.0

        return math.exp((energy - new_energy) / temp)

    """
    Generates a new solution candidate for simulated annealing. Neighbourhood function.
    Problem specific: Overwrite this method in a subclass.
    """
    def step(self, solution):
        raise NotImplementedError

    """
    Computes energy of given candidate solution
    Problem specific: Overwrite this method in a subclass.
    """
    def energy(self, solution):
        raise NotImplementedError


class TSPSAOptimizer(SimulatedAnnealingOptimizer):
    def step(self, solution):
        """
        # todo: we could also choose which elements to swap using a trained policy
        """
        new_solution = solution[:]
        a = random.randint(0, len(new_solution) - 1)
        b = random.randint(0, len(new_solution) - 1)
        new_solution[a] = new_solution[b]
        new_solution[b] = solution[a]

        return new_solution

    def energy(self, solution):
        distance = 0
        for i, city in enumerate(solution):
            if i < len(solution) - 1:
                distance += self.euclidean_distance(city, solution[i + 1])

        distance += self.euclidean_distance(solution[0], solution[-1])

        return distance

    def euclidean_distance(self, a, b):
        return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

if __name__ == '__main__':
    from envs.tsp.TSP import TSPGym
    tsp = TSPGym(num_cities=15)

    optimizer = TSPSAOptimizer()

    from stable_baselines3 import PPO

    model = PPO.load("ppo_tsp_sa")
    obs = tsp.reset()
    print(optimizer.optimize(tsp.state['unscheduled'][:]))

    done = False
    while not done:

        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = tsp.step(action)

    print(reward)
    print(optimizer.optimize(tsp.state['tour'][:]))

    tsp.render()
