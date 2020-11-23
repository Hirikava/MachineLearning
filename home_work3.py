import gym
import numpy as np

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
        self.states = np.full(
            (dimensions[0].number * dimensions[1].number * dimensions[2].number * dimensions[3].number, 2), 1)
        self.states = self.states / 2

    def get_state(self, observation):
        state = round((observation[0] + self.dimensions[0].offset) / self.dimensions[0].max * self.dimensions[0].number)
        state += round(sigma(observation[1]) * self.dimensions[1].number) * self.dimensions[0].number
        state += round(
            (observation[2] + self.dimensions[2].offset) / self.dimensions[2].max * self.dimensions[2].number) * \
                 self.dimensions[0].number * self.dimensions[1].number
        state += round(sigma(observation[3]) * self.dimensions[3].number) * self.dimensions[0].number * self.dimensions[
            1].number * self.dimensions[2].number
        return int(state)

    def make_action(self, state, epsilon):
        return get_epsilon_greedy_action(self.states[state], epsilon, 2)


    def update_policy(self):
        pass

    def get_session(self, session_length, environment, epsilon, render=False):
        observation = environment.reset()
        states, actions = list(), list()
        total_reward = 0

        for i in range(session_length):

            state = self.get_state(observation)
            states.append(state)
            action = self.make_action(state, epsilon)
            actions.append(action)

            observation, reward, done, _ = environment.step(action)
            total_reward += reward
            if render:
                environment.render()
            if done:
                break

        return {
            "states": states,
            "actions": actions,
            "total_reward": total_reward
        }


dimensions = (Dimension(9.6, 4.8, 10), Dimension(1, 0, 10), Dimension(0.936, 0.418, 10), Dimension(1, 0, 10))
agent = MonteCarloAgent(dimensions)

environment = gym.make("CartPole-v0")
environment.reset()

agent.get_session(100,environment,0.1,True);
