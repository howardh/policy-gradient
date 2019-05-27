import gym
import roboschool
import torch

import utils

"""
Actor-critic
"""

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))

#env = gym.make('RoboschoolReacher-v1')
env = gym.make('CartPole-v1')
env = utils.RecordingWrapper(env)
print(env.observation_space)
print(env.action_space)

theta_v = torch.rand([env.action_space.n*env.observation_space.shape[0],1])
theta_policy = torch.rand([env.action_space.n*env.observation_space.shape[0],1])

def features(obs, action):
    n = env.action_space.n
    m = env.observation_space.shape[0]
    output = torch.zeros([n*m,1])
    output[action*m:(action+1)*m,0] = torch.from_numpy(obs)
    return output

def policy(obs, theta):
    n = env.action_space.n
    pref = [theta.t().mm(features(obs,a)) for a in range(n)]
    pref = torch.tensor(pref)
    probs = torch.exp(pref)/torch.exp(pref).sum()
    return torch.distributions.Categorical(probs)

def value(obs, theta_v, theta_policy):
    p = policy(obs, theta_policy)
    return sum([x*theta_v.t().mm(features(obs,a)).item() for a,x in enumerate(p.probs)])

gamma = 0.9
learning_rate_v = 0.001
learning_rate_policy = 0.01

epoch = 0
while True:
    reward_history = []
    epoch += 1
    for iteration in range(30):
        episode = []
        done = False
        obs = env.reset()
        reward = None
        action = policy(obs,theta_policy).sample().item()
        episode.append((obs, reward, action))
        theta_v_change = 0
        theta_policy_change = 0

        if iteration == 0:
            env.record_to('videoframes/epoch-%05d'%epoch)

        i = 1
        while not done:
            obs, reward, done, info = env.step(action) # take a random action
            if not done:
                action = policy(obs,theta_policy).sample().item()
            else:
                action = None
            episode.append((obs, reward, action))

            if len(episode) > 1 and action is not None:
                prev_obs = episode[-2][0]
                delta = reward + gamma*value(obs,theta_v,theta_policy) - value(prev_obs,theta_v,theta_policy)
                theta_v_change += learning_rate_v*delta*features(obs,action)

                p = policy(obs,theta_policy).probs
                grad = features(obs,action) + sum([p[a]*features(obs,a) for a in range(env.action_space.n)])
                theta_policy_change += learning_rate_policy*i*delta*grad
                i *= gamma

        env.stop_recording()

        theta_policy += theta_policy_change
        theta_v += theta_v_change

        total_rewards = sum([r for _,r,_ in episode[1:]])
        reward_history.append(total_rewards)
    print('Epoch: %d \t Total rewards: %d' % (epoch, torch.mean(torch.tensor(reward_history))))

env.close()
