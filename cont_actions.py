import gym
import roboschool
import torch

import utils

"""
Basic implementation of REINFORCE with continuous actions.
Works, but quickly diverges.
"""

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))

#env = gym.make('RoboschoolReacher-v1')
env = gym.make('RoboschoolHopper-v1')
env = utils.RecordingWrapper(env)
print(env.observation_space)
print(env.action_space)

theta_mu = torch.rand([env.observation_space.shape[0],env.action_space.shape[0]])
theta_sigma = torch.rand([env.observation_space.shape[0],env.action_space.shape[0]])

def features(obs):
    return torch.from_numpy(obs).view(-1,1).float()

def policy(obs, theta_mu, theta_sigma):
    sigma = torch.exp(theta_sigma.t().mm(features(obs)))
    mu = theta_mu.t().mm(features(obs))
    return torch.distributions.Normal(mu, sigma)

gamma = 1
learning_rate = 1e-8

epoch = 0
while True:
    reward_history = []
    epoch += 1
    for iteration in range(10):
        episode = []
        done = False
        obs = env.reset()
        reward = None
        action = policy(obs,theta_mu,theta_sigma).sample()
        episode.append((obs, reward, action))
        frame_count = 0

        if iteration == 0:
            env.record_to('videoframes/epoch-%05d'%epoch)

        while not done:
            obs, reward, done, info = env.step(action.numpy())
            if not done:
                action = policy(obs,theta_mu,theta_sigma).sample()
            else:
                action = None
            episode.append((obs, reward, action))

        env.stop_recording()

        theta_mu_change = 0
        theta_sigma_change = 0
        for i,(obs,reward,action) in enumerate(episode[:-1]):
            mc_return = sum([(gamma**j)*r for j,(_,r,_) in enumerate(episode[i+1:])])
            p = policy(obs,theta_mu,theta_sigma)
            grad_mu = (1/(p.stddev**2)*(action-p.mean)).mm(features(obs).t()).t()
            grad_sigma = ((action-p.mean)**2/p.stddev**2 - 1).mm(features(obs).t()).t()
            theta_mu_change += learning_rate * (gamma**i) * mc_return * grad_mu
            theta_sigma_change += learning_rate * (gamma**i) * mc_return * grad_sigma
        theta_mu += theta_mu_change
        theta_sigma += theta_sigma_change

        total_rewards = sum([r for _,r,_ in episode[1:]])
        reward_history.append(total_rewards)
    print('Epoch: %d \t Total rewards: %d' % (epoch, torch.mean(torch.tensor(reward_history))))

env.close()
