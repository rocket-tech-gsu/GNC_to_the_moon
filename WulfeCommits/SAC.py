import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import os
import math 

#TODO the policy network should also have an option to ignite the rocket engine
#TODO do the output layers still need to be softmaxxed if theyre being used not as hot encodings?
#TODO make replay buffer from scratch
    #TODO replay buffer will have a look up method per state/action pair, 
    #TODO annealing method for low reward transitions once max size is met
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.ptr = 0  # Current position to write
        self.size = 0  # Current buffer size

        # Pre-allocate memory with float32 for efficiency
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.action_log_prob_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.base1_reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.trget1_reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.base2_reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.trget2_reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action,action_log_prob,reward,base_reward1, base_reward2, target_reward1, target_reward2,state_, done):
        index = self.ptr
        self.state_memory[index] = state
        self.new_state_memory[index] = state_  
        self.action_memory[index] = action
        self.action_log_prob_memory[index] = action_log_prob
        self.reward_memory[index] = reward
        self.base1_reward_memory[index] = base_reward1
        self.trget1_reward_memory[index] = target_reward1
        self.base2_reward_memory[index] = base_reward2
        self.trget2_reward_memory[index] = target_reward2
        self.terminal_memory[index] = done

        self.ptr = (self.ptr + 1) % self.mem_size
        self.size = min(self.size + 1, self.mem_size)

    def sample_buffer(self, batch_size):
        # Handle edge case where buffer has fewer samples than batch_size
        max_mem = min(self.size, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        action_log_prob = self.action_log_prob_memory[batch]
        rewards = self.reward_memory[batch]
        base1_rewards = self.base1_reward_memory[batch]
        target1_rewards = self.trget1_reward_memory[batch]
        base2_rewards = self.base2_reward_memory[batch]
        target2_rewards = self.trget2_reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, states_, actions, action_log_prob ,rewards ,base1_rewards, base2_rewards, target1_rewards ,target2_rewards ,dones
    

class QNetwork(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,beta,chkpt_dir='/Simulator'):
        super(QNetwork,self).__init__()

        #input dimensions should be action+state
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims
        self.beta = beta
        self.checkpoint_directory = chkpt_dir
        self.optimizer = optim.Adam(self.parameters,beta)

        self.checkpoint_file = os.path.join(chkpt_dir,'_sac')
        self.device = t.device('cpu')

        self.input_layer = nn.Linear(input_dims,fc1_dims)
        self.fc1 = nn.Linear(fc1_dims,fc2_dims)
        self.fc2 = nn.Linear(fc2_dims,output_dims)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))

    def forward(self,state,action):
        #TODO this input might need some dim= typecasting
        input=t.concatenate(state,action)
        currValue=self.input_layer(input)
        #consider using leaky relu for vanishing gradient
        currValue = F.relu(currValue)
        currValue = self.fc1(currValue)
        currValue = F.relu(currValue)
        currValue = self.fc2(currValue)
        currValue = F.softmax(currValue)

        return currValue


class PolicyNetwork(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,beta,chkpt_dir='/Simulator'):
        super(PolicyNetwork,self).__init__()
        self.max_angle = 20
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.beta = beta
        self.checkpoint_directory = chkpt_dir
        self.optimizer = optim.Adam(self.parameters,beta)

        self.checkpoint_file = os.path.join(chkpt_dir,'_sac')
        self.device = t.device('cpu')

        self.input_layer = nn.Linear(input_dims,fc1_dims)
        self.fc1 = nn.Linear(fc1_dims,fc2_dims)
        self.mean_output = nn.Linear(fc2_dims,1)
        self.std_output = nn.Linear(fc2_dims,1)
        #epsilon is for stochastic sampling of deterministic output
        self.epsilon = .3

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))

    def sampleAction(self,state):
        currValue=self.input_layer(state)
        #consider using leaky relu for vanishing gradient
        currValue = F.relu(currValue)
        currValue = self.fc1(currValue)
        currValue = F.relu(currValue)
        mean = self.mean_output(currValue)
        std = self.std_output(currValue)
        #TODO do these need to be softmaxxed?
        std=math.log(std)
        std=math.exp(std)*self.epsilon
        std = t.clip(std,min=-2,max=20)
        
        normal_dist = t.normal(mean,std)
        z = normal_dist.rsample()

        log_prob = normal_dist.log_prob(z)
    
    # Adjust for tanh squashing (log_prob correction)
        log_prob -= t.log(1 - t.tanh(z).pow(2) + 1e-6)

        action = t.tanh(z)
        scaled_action = action * self.max_angle
        #hadamard product needed here^^
        return scaled_action, log_prob


def main():
    training_iterations = 0
    training_batch_size = 1000
    n_iterations = None
    threshold = None

    Policy = PolicyNetwork(None,None,None,None)
    QNetwork_base1 = QNetwork(None,None,None,None,None)
    QNetwork_base2 = QNetwork(None,None,None,None,None)
    QNetwork_target1 = QNetwork(None,None,None,None,None)
    QNetwork_target2 = QNetwork(None,None,None,None,None)
    replay_buff = ReplayBuffer(None,None,None)
    #Q network has extra parameter for defining the output dimensions
    env = None
    terminated = False

    alpha = 0.1
    gamma = 0.1
    tau = 0.9
    actor_loss = 0
    base1_q_loss = 0
    base2_q_loss = 0
    target1_q_loss = 0
    target2_q_loss = 0
    #terminated will be an output from the environment DUH
    #env will be parameterized by the user's physics simulation

    while(n_iterations<threshold):
        #reset env

        while(not(terminated)):

            action = Policy.sampleAction(state=None)

            #policy output will be tanh between -1 and 1 for continous actions.
            #TODO this raises the question, how will the policy declaration differ between different simulations?

            target1_val=QNetwork_target1.forward(state,action)
            target2_val=QNetwork_target2.forward(state,action)
            base1_val = QNetwork_base1.forward(state,action)
            base2_val = QNetwork_base2.forward(state,action)
            stored_target_val = min(target1_val,target2_val)
            stored_base_val = min(base1_val,base2_val)
            #TODO do you need to store both the predicted q values in each replay buffer transition?
            
            env = None# state, reward, isTerminated= env.step()
            state_ = None
            reward = None
            state = None
            replay_buff.store_transition(None,None,None,None)
        n_iterations+=1
    
    while(training_iterations<training_batch_size):
        states, states_, actions, action_log_prob, reward ,base1_rewards, base2_rewards, target1_rewards ,target2_rewards ,dones = replay_buff.sample_buffer()
        base_rewards = min(base1_rewards,base2_rewards)
        actor_loss += (base_rewards -alpha * action_log_prob)

        next_action,next_log_prob = Policy.sampleAction(states_)
        target1_val = QNetwork_target1.forward(states_,next_action)
        target2_val = QNetwork_target2.forward(states_,next_action)

        stored_base_val = min(target1_val, target2_val)
        y = gamma*( 1 - int(dones))(stored_base_val - (alpha * next_log_prob))

        base1_q_loss += (y-base1_rewards)^2
        base2_q_loss += (y-base2_rewards)^2
        target1_q_loss += (tau * target1_rewards) + (1-tau)(base_rewards)
        target2_q_loss += (tau * target2_rewards) + (1-tau)(base_rewards)


    actor_loss/=training_batch_size
    base1_q_loss/=training_batch_size
    base2_q_loss/=training_batch_size
    target1_q_loss/=training_batch_size
    target2_q_loss/=training_batch_size

    Policy.optimizer.zero_grad()
    QNetwork_base1.optimizer.zero_grad()
    QNetwork_base2.optimizer.zero_grad()
    QNetwork_target1.optimizer.zero_grad()
    QNetwork_target2.optimizer.zero_grad()

    actor_loss.backward()
    base1_q_loss.backward()
    base2_q_loss.backward()
    target1_q_loss.backward()
    target2_q_loss.backward()

    QNetwork_base1.optimizer.step()
    QNetwork_base2.optimizer.step()
    QNetwork_target1.optimizer.step()
    QNetwork_target2.optimizer.step()

    #TODO add alpha annealing!!


    

    
        #y = reward + gamma(1-d)(Q_trg(s',a') -alog(s'|a'))
        #base networks update, mean of sum of: (y-q_b(s,a))^2
        #target q network update =(tau(q_trg) + (1-tau)(q_base))
        #
    
            #state,reward comes from the env DUH

            #store minQ_target, minQ_base, currstate, action, reward, nextState in the replay buffer



    #THESE COME FROM AT THE END OF THE LOOP, batch updates come from the replay buffer
    #target Q network output = reward + gamma(1-d)[{minQ_targ(s',a')}-alog(a'|s')]
    #target Q network loss/backprop update: [{t=.99}(Q_targ->r) + (1-t)(Q_base->r)]
    #base Q network loss/backprop update: (y-r)^2

    ##reparam notes down below

    #instead of resampling from pi_θ(`|s)
    #we choose to parameterize the action as a = f_θ(e,s)
    #where epsilon is a normal distribution from N(0,I) 
    # I is a randomized std, taking care of the stochasity whilst still making it differentiable
    
    #θ* = argmax(E_s~D)[E_a-pi('|s)][Q(s,a) - alog_piθ(a|s)]
    #the Q encourages actions taking the highest reward/prob 
    # with the self information promoting variability but alpha 

    #we replace the policy with the N dist prev mentioned
    #a_θ(s,e)=tanh(mean_θ(s)+std_θ(s)*e)
    #tanh ensures actions stay betweesn -1,1
    #tanh is simply chosen because it is some psuedo standard for action spaces
    #this will depend on our simulator I guess

    #policy loss = E_s~D,r~N(0,I)[alogpi_θ(a_θ(s,e)|s)-Q(s,a)]


    #these losses are randomly sampled from the replay buffer after some batch
    #batch length must also be defined

if __name__ == "__main__":
    main()