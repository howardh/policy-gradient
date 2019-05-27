import gym
import roboschool
import torch
import torch.utils.data
from tqdm import tqdm
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import RoboschoolHalfCheetah_v1_2017jul

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / torch.sqrt(torch.tensor(fanin).float())
    return torch.Tensor(size).uniform_(-v, v)

class QNetwork2(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(QNetwork2, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=400)
        self.fc2 = torch.nn.Linear(in_features=400,out_features=300)
        self.fc3 = torch.nn.Linear(in_features=300+100,out_features=1)
        self.fca = torch.nn.Linear(in_features=action_size,out_features=100)
        #self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=128)
        #self.fc2 = torch.nn.Linear(in_features=128,out_features=64)
        #self.fc3 = torch.nn.Linear(in_features=64+action_size,out_features=1)
        self.fca = torch.nn.Linear(in_features=action_size,out_features=100)
        self.relu = torch.nn.LeakyReLU()

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003,0.003)

    def forward(self, state, action):
        a = self.relu(self.fca(action))
        x = state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.cat([x,a],1)
        #x = torch.cat([x,a],1)
        x = self.fc3(x)
        return x

class QNetwork(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=400)
        self.fc2 = torch.nn.Linear(in_features=400+action_size,out_features=300)
        self.fc3 = torch.nn.Linear(in_features=300,out_features=1)
        self.relu = torch.nn.LeakyReLU()

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003,0.003)

    def forward(self, state, action):
        a = action
        x = state
        x = self.relu(self.fc1(x))
        x = torch.cat([x,a],1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        #self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=400)
        #self.fc2 = torch.nn.Linear(in_features=400,out_features=300)
        #self.fc3 = torch.nn.Linear(in_features=300,out_features=action_size)
        self.fc1 = torch.nn.Linear(in_features=obs_size,out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128,out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64,out_features=action_size)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-0.003,0.003)

    def forward(self, state, noise=0):
        x = state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        #x = self.tanh(self.fc3(x)+noise)
        x = (self.fc3(x)+noise).clamp(-1,1)
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
            transition = (obs0, action0, reward, obs, terminal)
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

batch_size = 64
min_replay_buffer_size = 10000
gamma = 0.99
tau = 0.001
critic_learning_rate = 1e-3
critic_weight_decay = 1e-2
actor_learning_rate = 1e-4
actor_weight_decay = 0

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#env = gym.make('RoboschoolHopper-v1')
#env = gym.make('RoboschoolReacher-v1')
#env = gym.make('RoboschoolAnt-v1')
#env = gym.make('RoboschoolInvertedPendulum-v1')
env = gym.make('RoboschoolHalfCheetah-v1')
#env = gym.make('MountainCarContinuous-v0')

critic = QNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)
critic_target = QNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)
actor = PolicyNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)
actor_target = PolicyNetwork(env.observation_space.shape[0],
        env.action_space.shape[0]).to(device)

#pretrained_actor = PolicyNetwork(env.observation_space.shape[0],
#        env.action_space.shape[0]).to(device)
#pretrained_actor.fc1.weight.data = RoboschoolHalfCheetah_v1_2017jul.weights_dense1_w.t().to(device)
#pretrained_actor.fc1.bias.data = RoboschoolHalfCheetah_v1_2017jul.weights_dense1_b.to(device)
#pretrained_actor.fc2.weight.data = RoboschoolHalfCheetah_v1_2017jul.weights_dense2_w.t().to(device)
#pretrained_actor.fc2.bias.data = RoboschoolHalfCheetah_v1_2017jul.weights_dense2_b.to(device)
#pretrained_actor.fc3.weight.data = RoboschoolHalfCheetah_v1_2017jul.weights_final_w.t().to(device)
#pretrained_actor.fc3.bias.data = RoboschoolHalfCheetah_v1_2017jul.weights_final_b.to(device)

gaussian_noise = torch.distributions.Normal(0,0.1)

for p1,p2 in zip(critic_target.parameters(), critic.parameters()):
    p1.data = p2.data
for p1,p2 in zip(actor_target.parameters(), actor.parameters()):
    p1.data = p2.data

replay_buffer = ReplayBuffer(int(1e6))
critic_optimizer = torch.optim.Adam(critic.parameters(),
        lr=critic_learning_rate, weight_decay=critic_weight_decay)
actor_optimizer = torch.optim.Adam(actor.parameters(),
        lr=actor_learning_rate, weight_decay=actor_weight_decay)

learning_curve = []
reward_history = []
episode_length = []
actor_grads = []
critic_grads = []
actor_loss_history = []
critic_loss_history = []
for episode in itertools.count():
    reward_history.append(0)
    episode_length.append(0)
    actor_grads.append([])
    critic_grads.append([])
    actor_loss_history.append([])
    critic_loss_history.append([])

    noise = gaussian_noise.sample(env.action_space.shape)

    done = False
    obs = env.reset()
    obs = torch.tensor(obs).float().to(device)
    reward = None
    #action = pretrained_actor(obs, noise.to(device)).detach()
    action = actor(obs, noise.to(device)).detach()
    replay_buffer.add_transition(obs, reward, action)

    while not done:
        episode_length[-1] += 1
        # Update noise
        noise = gaussian_noise.sample(env.action_space.shape)
        # Do stuff
        obs, reward, done, info = env.step(action.detach().cpu().numpy())
        obs = torch.tensor(obs).float().to(device)
        reward_history[-1] += reward
        if not done:
            #action = pretrained_actor(obs, noise.to(device)).detach()
            action = actor(obs, noise.to(device)).detach()
            replay_buffer.add_transition(obs, reward, action)
        else:
            replay_buffer.add_transition(obs, reward, None, terminal=True)

        # Train on replay buffer
        if len(replay_buffer) > min_replay_buffer_size:
            dataloader = torch.utils.data.DataLoader(replay_buffer, batch_size=batch_size, shuffle=True)
            for i,(s0,a0,r1,s1,t) in enumerate(dataloader):
                # Fix data types
                r1 = r1.float().view(-1,1).to(device)
                t = t.float().view(-1,1).to(device)
                # Value estimate
                y = r1+gamma*critic_target(s1,actor_target(s1))*(1-t)
                #y = r1+gamma*critic_target(s1,pretrained_actor(s1))*(1-t)
                #y = y.detach()
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
                # Save values for debugging
                for p in actor.parameters():
                    actor_grads[-1].append(torch.abs(p.grad).sum().item())
                for p in critic.parameters():
                    critic_grads[-1].append(torch.abs(p.grad).sum().item())
                actor_loss_history[-1].append(actor_loss.item())
                critic_loss_history[-1].append(critic_loss.item())

                # Update target weights
                for p1,p2 in zip(critic_target.parameters(), critic.parameters()):
                    p1.data = (1-tau)*p1+tau*p2
                for p1,p2 in zip(actor_target.parameters(), actor.parameters()):
                    p1.data = (1-tau)*p1+tau*p2
                #critic_target.load_state_dict(critic.state_dict())
                #actor_target.load_state_dict(actor.state_dict())

                break # One iteration only
    actor_grads[-1] = torch.tensor(actor_grads[-1]).float().mean()
    critic_grads[-1] = torch.tensor(critic_grads[-1]).float().mean()
    actor_loss_history[-1] = torch.tensor(actor_loss_history[-1]).float().mean()
    critic_loss_history[-1] = torch.tensor(critic_loss_history[-1]).float().mean()

    # Test
    if episode%1 == 0:
        test_rewards = []
        est_action_value = []
        sa_values = []
        for test_iteration in range(10):
            done = False
            obs = env.reset()
            obs = torch.tensor(obs).float().to(device)
            action = actor(obs).detach()

            test_rewards.append(0)
            est_action_value.append(critic(obs.view(1,*obs.shape),action.view(1,*action.shape)).item())

            while not done:
                sa_values.append(critic(obs.view(1,*obs.shape),action.view(1,*action.shape)).item())
                obs, reward, done, info = env.step(action.cpu().numpy())
                obs = torch.tensor(obs).float().to(device)
                test_rewards[-1] += reward
                if not done:
                    action = actor(obs).detach()

        # Plot learning curve
        learning_curve.append((episode, torch.tensor(test_rewards).mean().float()))
        plt.plot([x for x,y in learning_curve], [y for x,y in learning_curve])
        plt.xlabel('Episodes')
        #plt.ylabel('Test reward')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig('learningcurve.png')
        plt.close()

        print('Episode: %d \t Total rewards: %d \t Test rewards: %d \t Episode lengths: %d \t Initial State Value: %f \t AV: %f \t AL: %f \t CL: %f' % (episode,
            reward_history[-1], 
            torch.tensor(test_rewards).mean().float(),
            episode_length[-1],
            torch.tensor(est_action_value).float().mean(),
            torch.tensor(sa_values).float().mean(),
            #actor_grads[-1],
            #critic_grads[-1],
            actor_loss_history[-1],
            critic_loss_history[-1],
        ))

env.close()
