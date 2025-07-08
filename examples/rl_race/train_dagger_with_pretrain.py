import torch
import torch.nn as nn
import gym
import torch.optim as optim
from stable_baselines3 import PPO
from f110_gym.envs.base_classes import Integrator
import time
import numpy as np
from matplotlib import pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # [1, 64, 96]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [32, 32, 48]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [64, 16, 24]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [64, 8, 12]
            nn.Flatten(),
            nn.Linear(64 * 8 * 12, 128),
            nn.ELU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 64)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
            )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 64 * 8 * 12),
            nn.ELU(),
            nn.Unflatten(1, (64, 8, 12)),
            # [64, 8, 12]
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [64, 16, 24]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [32, 32, 48]
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        # print(h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

class DepthActor(nn.Module):
    def __init__(self):
        super().__init__()

        self.combination_mlp = nn.Sequential(
                                nn.Linear(64 + 6, 256),
                                nn.ELU(),
                                nn.Linear(256, 256)
                            )
        
        self.rnn = nn.GRUCell(input_size=256, hidden_size=256)
        self.hidden_states = None

        self.output_mlp = nn.Sequential(
                            nn.Linear(256, 64),
                            nn.ELU()
                        )
        self.actor = nn.Sequential(
                        nn.Linear(64, 256),
                        nn.ELU(),
                        nn.Linear(256, 256),
                        nn.ELU(),
                        nn.Linear(256, 2),
                        nn.Tanh())

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor):
        # depth_image = self.encoder(depth_image)
        # depth_image = self.fc1(depth_image)
        # depth_image = depth_image / 30.0
        proprioception = proprioception * 30.0
        x = torch.cat((depth_image, proprioception), dim=-1)
        depth_latent = self.combination_mlp(x)
        self.hidden_states = self.rnn(depth_latent, self.hidden_states)
        out_latent = self.output_mlp(self.hidden_states)
        output = self.actor(out_latent)
        return output

    def reset_hidden_states(self, hidden=None):
        self.hidden_states = hidden

class State_Action_Buffer:
    def __init__(self):
        self.teacher_actions = []
        self.teacher_obs = []
        self.depths = []
        self.dyn_states = []

    def put(self, teacher_action, obs, depth, dyn_state):
        self.teacher_actions.append(teacher_action.clone().cpu().numpy())
        self.teacher_obs.append(obs)
        self.depths.append(depth.clone().cpu().numpy())
        self.dyn_states.append(dyn_state.clone().cpu().numpy())

    def sample(self, idx, lookup_step):
        teacher_actions = self.teacher_actions[idx:idx+lookup_step]
        teacher_obs = self.teacher_obs[idx:idx+lookup_step]
        depths = self.depths[idx:idx+lookup_step]
        dyn_states = self.dyn_states[idx:idx+lookup_step]
        return dict(teachers=teacher_actions,
                    obs=teacher_obs,
                    deps=depths,
                    dyns=dyn_states)
    
    def __len__(self):
        return len(self.teacher_actions)

class Episode_Buffer:
    def __init__(self, buffer_size, lookup_step):
        self.buffer = []
        self.size = buffer_size
        self.lookup_step = lookup_step

    def put(self, episode):
        self.buffer.append(episode)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def clear(self):
        self.buffer.clear()

    def buffer_size(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        sampled_buffer = []
        sampled_episodes = random.sample(self.buffer, batch_size)
        min_step = 1024
        for episode in sampled_episodes:
            min_step = min(min_step, len(episode))
        for episode in sampled_episodes:
            if min_step > self.lookup_step:
                idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                sample = episode.sample(idx, self.lookup_step)
                sampled_buffer.append(sample)
            else:
                idx = np.random.randint(0, len(episode)-min_step+1)
                sample = episode.sample(idx, min_step)
                sampled_buffer.append(sample)

        teachers = []
        obs = []
        deps = []
        dyns = []
        for traj in sampled_buffer:
            teachers.append(np.array(traj['teachers']))
            obs.append(np.array(traj['obs']))
            deps.append(np.array(traj['deps']))
            dyns.append(np.array(traj['dyns']))
        teachers = np.array(teachers)
        obs = np.array(obs)
        deps = np.array(deps)
        dyns = np.array(dyns)

        return dict(teachers=teachers,
                    obs=obs,
                    deps=deps,
                    dyns=dyns), teachers.shape[1]

class DAgger:
    def __init__(self, env: gym.Env, teacher):
        self.env = env
        self.teacher = teacher
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.vae = VAE().to(self.device)
        param = torch.load('vae.pth')
        self.vae.encoder.load_state_dict(param['encoder'])
        self.vae.fc1.load_state_dict(param['fc1'])
        self.vae.fc2.load_state_dict(param['fc2'])
        self.vae.decoder.load_state_dict(param['decoder'])
        self.vae.eval()
        self.depth_actor = DepthActor().to(self.device)
        self.depth_actor_optimizer = optim.Adam(self.depth_actor.parameters(), lr=3e-4)
        self.batch_size = 32
        self.buffer_size = 128
        self.pretrain_iter = 0
        self.episode_per_epoch = 16
        self.train_per_epoch = 8
        self.max_episode_reward = -np.inf
        self.step_num = 0
        self.epoch_num = 0
        self.writer = SummaryWriter(log_dir='./runs/student_with_pretrain')

    def learn_vision(self, num_learning_iterations):
        self.depth_actor.train()
        all_reward = 0
        data_buffer = Episode_Buffer(buffer_size=self.buffer_size, lookup_step=128)

        for it in range(num_learning_iterations):
            # if it < 50:
            #     self.depth_actor_optimizer = optim.Adam(self.depth_actor.parameters(), lr=3e-4)
            # else:
            #     self.depth_actor_optimizer = optim.Adam(self.depth_actor.parameters(), lr=1e-4)
            self.depth_actor_optimizer = optim.Adam(self.depth_actor.parameters(), lr=3e-4)
            for i in range(self.episode_per_epoch):
                s_a_buffer = State_Action_Buffer()
                done = False
                teacher_obs = self.env.reset()
                student_obs = self.env.get_observation()
                self.depth_actor.reset_hidden_states(torch.zeros(1, 256).to(self.device))
                all_reward = 0
                while not done:
                    dep = student_obs['depth'].clone()
                    body = student_obs['body'].clone()
                    action_teacher, _states = self.teacher.predict(teacher_obs, deterministic=True)
                    with torch.no_grad():
                        action_teacher = torch.from_numpy(action_teacher).to(self.device).unsqueeze(0)
                    depth_latent, _, _ = self.vae.encode(dep.unsqueeze(0))
                    depth_latent = depth_latent.detach()
                    action_student = self.depth_actor(depth_latent, body)
                    if torch.isnan(action_student).all():
                        s_a_buffer = State_Action_Buffer()
                        break
                    s_a_buffer.put(action_teacher, teacher_obs, dep, body)
                    if it >= self.pretrain_iter:
                        teacher_obs, reward, done, info = self.env.step(action_student.detach().cpu().squeeze().numpy())
                    else:
                        teacher_obs, reward, done, info = self.env.step(action_teacher.detach().cpu().squeeze().numpy())
                    all_reward += reward
                    self.step_num += 1
                    student_obs = self.env.get_observation()
                if len(s_a_buffer) > 0:
                    data_buffer.put(s_a_buffer)

            if data_buffer.buffer_size() < self.batch_size:
                print("collecting data...")
            else:
                self.epoch_num += 1
                for i in range(self.train_per_epoch):
                    train_data, n_seq = data_buffer.sample(self.batch_size)
                    loss = self.update_depth_actor(train_data, n_seq, self.batch_size)
                    print("iter:", it, "loss:", loss)
                self.writer.add_scalar('loss', loss, self.epoch_num)
            if it % 5 == 1:    
                self.eval_actor()
                    
    def update_depth_actor(self, train_data, n_seq, batch_size):

        actions_teacher_buffer = []
        actions_student_buffer = []
        hidden = torch.zeros(batch_size, 256).to(self.device)
        self.depth_actor.reset_hidden_states(hidden)

        for t in range(n_seq):
            action_teacher = torch.from_numpy(train_data['teachers'][:, t, :]).float().to(self.device).squeeze(1)
            dep = torch.from_numpy(train_data['deps'][:, t, :]).float().to(self.device)
            dyn = torch.from_numpy(train_data['dyns'][:, t, :]).float().to(self.device).squeeze(1)
            depth_latent, _, _ = self.vae.encode(dep)
            depth_latent = depth_latent.detach()
            action_student = self.depth_actor(depth_latent, dyn)
            actions_teacher_buffer.append(action_teacher.clone())
            actions_student_buffer.append(action_student.clone())

        actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
        actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
        loss = nn.functional.mse_loss(actions_teacher_buffer, actions_student_buffer)
        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.depth_actor.parameters(), 1.0)
        self.depth_actor_optimizer.step()
        return loss.item()
    
    def load_model(self, path):
        self.depth_actor.load_state_dict(torch.load(path))

    def eval_actor(self):
        self.depth_actor.eval()
        everage_reward = 0
        N = 40
        self.env.max_time = 30
        for i in range(N):
            self.env.reset()
            student_obs = self.env.get_observation()
            self.depth_actor.reset_hidden_states(torch.zeros(1, 256).to(self.device))
            all_reward = 0
            done = False
            while not done:
                dep = student_obs['depth'].clone()
                body = student_obs['body'].clone()
                depth_latent, _, _ = self.vae.encode(dep.unsqueeze(0))
                depth_latent = depth_latent.detach()
                action_student = self.depth_actor(depth_latent, body)
                teacher_obs, reward, done, info = self.env.step(action_student.detach().cpu().squeeze().numpy())
                student_obs = self.env.get_observation()
                all_reward += reward
            everage_reward += all_reward
        self.env.max_time = 10 
        everage_reward /= float(N)
        if everage_reward > self.max_episode_reward:
            self.max_episode_reward = everage_reward
            torch.save(self.depth_actor.state_dict(), './student_model/depth_actor_with_pretrain.pth')
        self.writer.add_scalar('epoch_reward', everage_reward, self.epoch_num)
        self.writer.add_scalar('step_reward', everage_reward, self.step_num)
        self.depth_actor.train()

    def test_actor(self):
        self.depth_actor.eval()
        N = 40
        self.env.max_time = 50
        lap_times = []
        mean_jerks = []
        finished_progress = []
        success_count = 0
        for i in range(N):
            self.env.reset()
            student_obs = self.env.get_observation()
            self.depth_actor.reset_hidden_states(torch.zeros(1, 256).to(self.device))
            done = False
            traj = []
            start_process = env.last_process
            while not done and env.lap_counts[0] <= 0:
                dep = student_obs['depth'].clone()
                body = student_obs['body'].clone()
                depth_latent, _, _ = self.vae.encode(dep.unsqueeze(0))
                depth_latent = depth_latent.detach()
                action_student = self.depth_actor(depth_latent, body)
                teacher_obs, reward, done, info = self.env.step(action_student.detach().cpu().squeeze().numpy())
                student_obs = self.env.get_observation()
                traj.append(np.array([env.obs['poses_x'][0], env.obs['poses_y'][0]]))
                if env.last_process > start_process:
                    fake = env.last_process - start_process
                else:
                    fake = env.last_process + (env.max_process - start_process)
                fake = fake / env.max_process
                if fake > 0.99:
                    fake = 1.0
            if done is False:
                print("lap time =", env.current_time)
                lap_times.append(env.current_time)
                success_count += 1
                mean_jerks.append(self.calculate_mean_jerk(traj))
            finished_progress.append(fake)
        print(lap_times, np.mean(np.array(lap_times)))
        print(mean_jerks)
        print(finished_progress)
        print('success_rate', success_count / N)

    def calculate_mean_jerk(self, traj, dt=1.0/30.0):
        vel = []
        for i in range(len(traj)-1):
            v = (traj[i+1] - traj[i]) / dt
            vel.append(v)
        acc = []
        for i in range(len(vel)-1):
            a = (vel[i+1] - vel[i]) / dt
            acc.append(a)
        jerk = []
        for i in range(len(acc)-1):
            j = (acc[i+1] - acc[i]) / dt
            jerk.append(j)
        jerk_norm = []
        for i in range(len(jerk)):
            jerk_norm.append(np.linalg.norm(jerk[i]))
        mean_jerk = np.mean(np.array(jerk_norm))
        return mean_jerk




    
if __name__ == '__main__':
    map_path = './examples/rl_race/f1tenth_racetracks/ex1/ex'
    map_ext = '.png'
    wp_path = './examples/rl_race/f1tenth_racetracks/ex1/ex_centerline.npy'
    sdf_path = './examples/rl_race/f1tenth_racetracks/ex1/ex.sdf'

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

    freq = 30
    dt = 1.0 / (freq * 10.0)

    env = gym.make('f110_gym:f110-v0', 
                   map=map_path, 
                   map_ext=map_ext, 
                   waypoint=wp_path, 
                   sdf_path=sdf_path, 
                   num_agents=1, 
                   timestep=dt, 
                   integrator=Integrator.RK4, 
                   eval_flag=0, 
                   depth_render=1,
                   max_time=10)
    env.add_render_callback(render_callback)
    ppo_path = './best_model.zip'
    expert = PPO.load(ppo_path)
    dagger_trainer = DAgger(env, expert)

    



