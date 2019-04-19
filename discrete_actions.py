import gym
import roboschool
import torch

"""
Basic implementation of REINFORCE.
It learns, but diverges pretty quickly.
"""

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))

#env = gym.make('RoboschoolReacher-v1')
env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.action_space)

theta = torch.rand([env.action_space.n*env.observation_space.shape[0],1])

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

gamma = 0.9
learning_rate = 0.01

while True:
    reward_history = []
    for _ in range(30):
        episode = []
        done = False
        obs = env.reset()
        reward = None
        action = policy(obs,theta).sample().item()
        episode.append((obs, reward, action))
        while not done:
            obs, reward, done, info = env.step(action) # take a random action
            if not done:
                action = policy(obs,theta).sample().item()
            else:
                action = None
            episode.append((obs, reward, action))

        theta_change = 0
        for i,(obs,reward,action) in enumerate(episode[:-1]):
            mc_return = sum([(gamma**j)*r for j,(_,r,_) in enumerate(episode[i+1:])])
            p = policy(obs,theta).probs
            grad = features(obs,action)
            for a in range(env.action_space.n):
                grad += p[a]*features(obs,a)
            theta_change += learning_rate * (gamma**i) * mc_return * grad
        theta += theta_change

        total_rewards = sum([r for _,r,_ in episode[1:]])
        reward_history.append(total_rewards)
    print('Total rewards', torch.mean(torch.tensor(reward_history)))

env.close()
