import gym
import numpy as np
import math
from matplotlib import pyplot as pl

e = 2.7182818284
def sigma(x):
    return 1 / (1 + pow(e, -x))

def get_epsilon_greedy_action(q_values, epsilon, number_of_actions):
    prob = np.ones(number_of_actions) * epsilon / number_of_actions
    argmax_prob = np.argmax(q_values)
    prob[argmax_prob] += 1 - epsilon
    return np.random.choice(np.arange(number_of_actions), p=prob)

class Dimension:
    def __init__(self, max, offset, number):
        self.max = max
        self.offset = offset
        self.number = number


class MonteCarloAgent:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.number_of_states = dimensions[0].number * dimensions[1].number * dimensions[2].number * dimensions[3].number
        self.states = np.zeros((self.number_of_states,2))

    def get_state(self, observation):
        state = math.floor((observation[0] + self.dimensions[0].offset) / self.dimensions[0].max * self.dimensions[0].number)
        state += math.floor((observation[1] + self.dimensions[1].offset) / self.dimensions[1].max * self.dimensions[1].number) * self.dimensions[0].number
        state += math.floor(
            (observation[2] + self.dimensions[2].offset) / self.dimensions[2].max * self.dimensions[2].number) * \
                 self.dimensions[0].number * self.dimensions[1].number
        state += math.floor((observation[3] + self.dimensions[3].offset) / self.dimensions[3].max * self.dimensions[3].number) * self.dimensions[0].number * self.dimensions[
            1].number * self.dimensions[2].number
        return int(state)

    def make_action(self, state, epsilon):
        return get_epsilon_greedy_action(self.states[state], epsilon, 2)


    def update_policy(self, session, gamma):
        N = np.zeros((self.number_of_states, 2))
        G = list()

        d_gamma = gamma
        G.append(session["rewards"][-1])
        for reward in session["rewards"][-1::-1]:
            G.append(reward * d_gamma + G[-1])
            d_gamma *= gamma
        G.reverse()

        for t in range(len(session["actions"])):
            state = session["states"][t]
            action = session["actions"][t]
            self.states[state][action] += (G[t] - self.states[state][action]) / (1 + N[state][action])
            N[state][action] += 1


    def get_session(self, session_length, environment, epsilon, render=False):
        observation = environment.reset()
        states, actions, rewards, v1, v2 = list(), list(), list(), list(), list()

        for i in range(session_length):
            v1.append(observation[1])
            v2.append(observation[2])
            state = self.get_state(observation)
            action = self.make_action(state, epsilon)
            observation, reward, done, _ = environment.step(action)
            if render:
                environment.render()
            if done:
                break
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "total_reward": sum(rewards)

        }

dis = 7
dimensions = (Dimension(9.6, 4.8, dis), Dimension(10, 5, dis*2), Dimension(0.936, 0.418, dis*4), Dimension(10, 5, dis*2))
agent = MonteCarloAgent(dimensions)
environment = gym.make("CartPole-v0")

episode_count = 2000
clear_episode_count = 100
rewards = list()
session_length = 200

epsilon = 1
d_epsilon = epsilon / episode_count

for i in range(episode_count):
    print(str(i) + '/' + str(episode_count))
    session = agent.get_session(session_length, environment, epsilon)
    agent.update_policy(session, 0.99)
    epsilon = max(epsilon - d_epsilon, 0)
    rewards.append(session["total_reward"])

for i in range(clear_episode_count):
    print(str(i) + '/' + str(clear_episode_count) + " clear")
    session = agent.get_session(session_length, environment, 0)
    #agent.update_policy(session, 0.99)
    rewards.append(session["total_reward"])

plot = pl.plot(range(episode_count + clear_episode_count), rewards)
pl.show()

for i in range(10):
    session = agent.get_session(session_length, environment, 0, True)
    agent.update_policy(session, 0.99)
    rewards.append(session["total_reward"])
