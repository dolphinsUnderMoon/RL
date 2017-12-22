import numpy as np
import pandas as pd
import time

np.random.seed(2)


class config:
    num_states = 10
    actions = ['left', 'right']
    epsilon = 0.9
    leaning_rate = 0.1
    discount_factor = 0.9
    max_episodes = 20
    fresh_time = 0.3
    leaning_algorithm = 'sarsa'


def build_q_table(num_states, actions):
    q_table = pd.DataFrame(
        np.zeros((num_states, len(actions))),
        columns=actions
    )
    return q_table


def choose_actions(current_state, q_table):
    if current_state == 'terminal':
        current_state = config.num_states - 1
    alternate_actions = q_table.iloc[current_state, :]

    if np.random.uniform() > config.epsilon or alternate_actions.all() == 0:
        action_selection = np.random.choice(config.actions)
    else:
        action_selection = alternate_actions.idxmax()

    return action_selection


def get_environment_feedback(current_state, action_selection):
    if action_selection == 'right':
        if current_state == config.num_states - 2:
            next_state = 'terminal'
            reward = 1
        else:
            if current_state == 'terminal':
                next_state = current_state
                reward = 1
            else:
                next_state = current_state + 1
                reward = 0
    else:
        reward = 0
        if current_state == 0:
            next_state = current_state
        else:
            next_state = current_state - 1

    return next_state, reward


def update_environment(state, episode, step_counter):
    environment_visual = ['-'] * (config.num_states - 1) + ['T']
    if state == 'terminal':
        interaction = 'Episode: [%s]: total steps spending: [%s]' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
    else:
        environment_visual[state] = 'o'
        interaction = ''.join(environment_visual)
        print('\r{}'.format(interaction), end='o')
        time.sleep(config.fresh_time)


def rl_q_learning():
    q_table = build_q_table(config.num_states, config.actions)
    for episode in range(config.max_episodes):
        step_counter = 0
        current_state = 0
        is_terminated = False
        update_environment(current_state, episode, step_counter)

        while not is_terminated:
            action_selection = choose_actions(current_state, q_table)
            next_state, reward = get_environment_feedback(current_state, action_selection)
            q_last_time = q_table.ix[current_state, action_selection]

            if next_state != 'terminal':
                q_learn_value = reward + config.discount_factor * q_table.iloc[next_state, :].max()
            else:
                q_learn_value = reward
                is_terminated = True

            q_table.ix[current_state, action_selection] += config.leaning_rate * (q_learn_value - q_last_time)
            current_state = next_state

            step_counter += 1
            update_environment(current_state, episode, step_counter)

    return q_table


def rl_sarsa():
    q_table = build_q_table(config.num_states, config.actions)
    for episode in range(config.max_episodes):
        step_counter = 0
        current_state = 0
        is_terminated = False
        update_environment(current_state, episode, step_counter)
        action_selection = choose_actions(current_state, q_table)

        while not is_terminated:
            next_state, reward = get_environment_feedback(current_state, action_selection)
            action_ = choose_actions(next_state, q_table)

            q_last_time = q_table.ix[current_state, action_selection]

            if next_state != 'terminal':
                q_learn_value = reward + config.discount_factor * q_table.ix[next_state, action_]
            else:
                q_learn_value = reward
                is_terminated = True

            q_table.ix[current_state, action_selection] += config.leaning_rate * (q_learn_value - q_last_time)
            current_state = next_state
            action_selection = action_

            step_counter += 1
            update_environment(current_state, episode, step_counter)

    return q_table


if __name__ == "__main__":
    q_table_trained = None
    if config.leaning_algorithm == 'q-learning':
        q_table_trained = rl_q_learning()
    if config.leaning_algorithm == 'sarsa':
        q_table_trained = rl_sarsa()
    print("\nThe trained Q-table:")
    print(q_table_trained)
