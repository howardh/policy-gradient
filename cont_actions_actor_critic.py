import gym
import roboschool
import torch

import utils

"""
Actor-critic
"""

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))

env = gym.make('RoboschoolHopper-v1')
#env = gym.make('RoboschoolAnt-v1')
env = utils.RecordingWrapper(env)
print(env.observation_space)
print(env.action_space)

theta_v = torch.rand([env.observation_space.shape[0],1])
theta_mu = torch.rand([env.observation_space.shape[0],env.action_space.shape[0]])
theta_sigma = torch.rand([env.observation_space.shape[0],env.action_space.shape[0]])

def features(obs):
    return torch.from_numpy(obs).view(-1,1).float()

def policy(obs, theta_mu, theta_sigma):
    sigma = torch.exp(theta_sigma.t().mm(features(obs)))
    mu = theta_mu.t().mm(features(obs))
    return torch.distributions.Normal(mu, sigma)

def value(obs, theta_v):
    return theta_v.t().mm(features(obs)).item()

gamma = 1
learning_rate_v = 1e-3
learning_rate_mu = 1e-4
learning_rate_sigma = 1e-3

epoch = 0
while True:
    reward_history = []
    epoch += 1
    for iteration in range(10):
        episode = []
        done = False
        obs = env.reset()
        reward = None
        action = policy(obs, theta_mu, theta_sigma).sample().clamp(min=-1,max=1)
        episode.append((obs, reward, action))
        theta_v_change = 0
        theta_mu_change = 0
        theta_sigma_change = 0

        if iteration == 0:
            env.record_to('videoframes/epoch-%05d'%epoch)

        i = 1
        while not done:
            obs, reward, done, info = env.step(action.numpy())
            #if reward < -1: #clamp rewards
            #    reward = -1
            if not done:
                if torch.rand(1) < 0.0:
                    action = policy(obs,theta_mu,theta_sigma).sample().clamp(min=-1,max=1)
                else:
                    action = policy(obs,theta_mu,theta_sigma).mean
            else:
                action = None
            episode.append((obs, reward, action))

            if len(episode) > 1 and action is not None:
                prev_obs = episode[-2][0]
                delta = reward + gamma*value(obs,theta_v) - value(prev_obs,theta_v)
                theta_v_change += learning_rate_v*delta*features(obs)

                p = policy(obs,theta_mu,theta_sigma)
                grad_mu = (1/(p.stddev**2)*(action-p.mean)).mm(features(obs).t()).t()
                grad_sigma = ((action-p.mean)**2/p.stddev**2 - 1).mm(features(obs).t()).t()
                theta_mu_change += learning_rate_mu*i*delta*grad_mu
                theta_sigma_change += learning_rate_sigma*i*delta*grad_sigma
                i *= gamma

        env.stop_recording()

        assert(torch.isfinite(theta_v_change).all())
        assert(torch.isfinite(theta_mu_change).all())
        assert(torch.isfinite(theta_sigma_change).all())

        theta_v += theta_v_change
        theta_mu += theta_mu_change
        theta_mu = theta_mu.clamp(min=-1,max=1)
        theta_sigma += theta_sigma_change

        assert(torch.isfinite(theta_v_change).all())
        assert(torch.isfinite(theta_mu_change).all())
        assert(torch.isfinite(theta_sigma_change).all())

        last_change = (theta_v_change,theta_mu_change,theta_sigma_change)

        total_rewards = sum([r for _,r,_ in episode[1:]])
        reward_history.append(total_rewards)
    print('Epoch: %d \t Total rewards: %d' % (epoch, torch.mean(torch.tensor(reward_history).float())))

env.close()
