import gym
import roboschool
import torch
import torch.utils.data

import utils

class QNetwork(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(QNetwork, self).__init__()
        input_size = obs_size+action_size
        self.fc1 = torch.nn.Linear(in_features=input_size,out_features=int(input_size/2))
        self.fc2 = torch.nn.Linear(in_features=int(input_size/2),out_features=1)
        self.relu = torch.nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state,action],1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=int((obs_size+action_size)/2))
        self.fc2 = torch.nn.Linear(in_features=int((obs_size+action_size)/2),out_features=action_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, state, noise=0):
        x = state
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x)+noise)
        return x

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.index = 0
        self.prev_transition = None

    def add_transition(self, obs, reward, action, terminal=False):
        if self.prev_transition is not None:
            obs0, reward0, action0 = self.prev_transition
            transition = (obs0, action0, reward, obs)
            if len(self.buffer) < self.max_size:
                self.buffer.append(transition)
            else:
                self.buffer[self.index] = transition
                self.index = (self.index+1)%self.max_size
        if terminal:
            self.prev_transition = None
        else:
            self.prev_transition = (obs, reward, action)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

assert False

batch_size = 50
gamma = 1
tau = 0.1
critic_learning_rate = 0.001
actor_learning_rate = 0.001
device = torch.device('cpu')
#device = torch.device('cuda:0')

env = gym.make('RoboschoolHopper-v1')
env = utils.RecordingWrapper(env)

critic = QNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)
critic_target = QNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)
actor = PolicyNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)
actor_target = PolicyNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)

noise = torch.distributions.Normal(0,1)

for p1,p2 in zip(critic_target.parameters(), critic.parameters()):
    p1[:] = p2
for p1,p2 in zip(actor_target.parameters(), actor.parameters()):
    p1[:] = p2

replay_buffer = ReplayBuffer(10000)
critic_optimizer = torch.optim.SGD(critic.parameters(), lr=critic_learning_rate, momentum=0)
actor_optimizer = torch.optim.SGD(actor.parameters(), lr=actor_learning_rate, momentum=0)

epoch = 0
while True:
    reward_history = []
    epoch += 1
    for iteration in range(10):
        reward_history.append(0)
        done = False
        obs = env.reset()
        obs = torch.tensor(obs).to(device)
        reward = None
        action = actor(obs, noise.sample_n(env.action_space.shape[0]).to(device)).detach()
        replay_buffer.add_transition(obs, reward, action)

        while not done:
            obs, reward, done, info = env.step(action.detach().cpu().numpy())
            obs = torch.tensor(obs).to(device)
            reward_history[-1] += reward
            if not done:
                action = actor(obs, noise.sample_n(env.action_space.shape[0]).to(device)).detach()
                replay_buffer.add_transition(obs, reward, action)
            else:
                replay_buffer.add_transition(obs, reward, None, terminal=True)

        # Train on replay buffer
        if len(replay_buffer) > 1000:
            dataloader = torch.utils.data.DataLoader(replay_buffer, batch_size=batch_size, shuffle=True)
            for i,(s0,a0,r0,s1) in enumerate(dataloader):
                if i > 50:
                    break
                # Fix data types
                r0 = r0.float().view(-1,1).to(device)
                # Value estimate
                y = r0+gamma*critic_target(s1,actor_target(s1))
                y = y.detach()
                # Update critic
                critic_optimizer.zero_grad()
                critic_loss = ((y-critic(s0,a0))**2).mean()
                critic_loss.backward()
                critic_optimizer.step()
                # Update actor
                actor_optimizer.zero_grad()
                actor_loss = -critic(s0,actor(s0)).mean()
                actor_loss.backward()
                actor_optimizer.step()

                # Update target weights
                for p1,p2 in zip(critic_target.parameters(), critic.parameters()):
                    p1[:] = (1-tau)*p1+tau*p2
                for p1,p2 in zip(actor_target.parameters(), actor.parameters()):
                    p1[:] = (1-tau)*p1+tau*p2

    # Test
    test_rewards = []
    for iteration in range(10):
        test_rewards.append(0)
        done = False
        obs = env.reset()
        obs = torch.tensor(obs).to(device)
        action = actor(obs).detach()

        while not done:
            obs, reward, done, info = env.step(action.detach().cpu().numpy())
            obs = torch.tensor(obs).to(device)
            test_rewards[-1] += reward
            if not done:
                action = actor(obs).detach()

    print('Epoch: %d \t Total rewards: %d \t Test rewards: %d' % (epoch,
        torch.tensor(reward_history).mean().float(), 
        torch.tensor(test_rewards).mean().float()))

env.close()
