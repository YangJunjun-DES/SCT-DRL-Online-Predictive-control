import os
os.environ["OMP_NUM_THREADS"] = '1'


import scipy.io as sio
import numpy as np
from MystepFun import AGV_StepFun
from param_class import AGV_param
from util import AGV_norm_state
import random
from RL_brain_class import DeepQNetwork


from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # 关闭 2.x 行为



# %% Load models
plant_param = AGV_param()
# %% hyperparameters defination
random.seed(5) # 50

NUM_ACTION = 10 #Number of controllable events
NUM_OBS = 7
MEMORY_SIZE = np.exp2(13).astype(int)
BATCH_SIZE = np.exp2(11).astype(int)
NUM_EPISODE = 50000
# initial state
INIT_OBS = [0, 0, 0, 0, 0, 0, 0]

MAX_EPI_STEP = 200
RECORD_VAL = 600

#Online module
STEP = 1    #0 means no online improvement
            #1 means 1-step lookahead
#2 means rollout
RO_NODES = 10
RO_TRACES = RO_NODES*2
RO_DEPTH = 5
RO_gamma = 0.9


#Cluster module
Cluster = 1
K = 2
Delta = 0.6
            #1             #2  (1:lookahead,2:rollout)
#0.05       3843
#0.15       3839
#0.2        3842
#0.3        4205
#0.35       4383
#0.4        4405
#0.5        4406
#0.6        4406

Single_model = 1  # 1 Use single-model
                    # 0 Use multi-model

# train or not    
Train = 0      # 0:Check
               # 1:Train

# build network
tf.reset_default_graph()
RL = DeepQNetwork(NUM_ACTION, 
                  NUM_OBS,
                  learning_rate = 1e-3,
                  reward_decay = 0.98,
                  e_greedy = 0.95,
                  replace_target_iteration = 100,
                  memory_size = MEMORY_SIZE,
                  batch_size = BATCH_SIZE,
                  epsilon_increment = 1e-4, 
                  epsilon_init = 0.10, 
                  output_graph = False,
                  max_num_nextS = 26,
                  l1_node = 128,  #128
                  l2_node = 128,
                  look_ahead_step = STEP,
                  RO_nodes = RO_NODES,
                  RO_traces = RO_TRACES,
                  RO_depth = RO_DEPTH,
                  RO_gamma = RO_gamma)

saver = tf.compat.v1.train.Saver(max_to_keep=None)
cwd = os.getcwd() + '\\' + datetime.today().strftime('%Y-%m-%d') + '\\AGV'
if not os.path.exists(cwd):
    os.makedirs(cwd)

total_step = 0
reward_history = []
good_event_history = []
episode_step_history = [0]
max_epi_reward = -100




# %% train
if Train:
    for num_episode in range(NUM_EPISODE):
        
        S = INIT_OBS # INIT_obs[random.randint(0, len(INIT_obs)-1)]
        init_S = S
        S_norm, _ = AGV_norm_state(S)
        episode_reward = 0
        episode_step = 0
        epi_good_event = 0
        epi_action_list = []
        if num_episode > 140000 and num_episode < 150000:
            RL.learning_rate -= 2e-8
        while True:         
            # initialize the Action
            A = RL.choose_action(S_norm, 1)
            # take action and observe
            [S_, all_S_, R, isDone, IfAppear32, stop_ind, selected_action] = \
                AGV_StepFun(S, A, plant_param)
            S_norm_, _ = AGV_norm_state(S_)
            all_S_norm_, _ = AGV_norm_state(all_S_)
            
            # store transition
            RL.store_exp(S_norm, A, R, all_S_norm_)
            # control the learning starting time and frequency
            if total_step > MEMORY_SIZE and (total_step % 10 == 0):
                RL.learn()
            # update states
            episode_reward += R
            episode_step += 1
            epi_good_event += IfAppear32
            S = S_
            S_norm = S_norm_
            if isDone == 1 or episode_step > MAX_EPI_STEP:
                if stop_ind == 1:
                    stop_reason = 'Undefined Controllable events'
                elif stop_ind == 2:
                    stop_reason = 'Deadlocks'
                elif episode_step > MAX_EPI_STEP:
                    stop_reason = 'reach 200 steps'
                else:
                    stop_reason = 'next state is empty'
                if max_epi_reward < episode_reward:
                    max_epi_reward = episode_reward
                print('episode:', num_episode, '\n', 
                      'init state:', init_S, '\n',
                      'episode reward:', episode_reward, '\n',
                      'episode step:', episode_step, '\n',
                      'good event:', epi_good_event, '\n',
                      'epsilon value:', RL.epsilon, '\n',
                      'action list:', epi_action_list, '\n',
                      'maximal running step:', np.max(episode_step_history), '\n',
                      'maximal episode reward:', max_epi_reward, '\n',
                      'total good event:', np.sum(good_event_history), '\n',
                      stop_reason, '\n',
                      '*******************************************')
                reward_history.append(episode_reward)
                good_event_history.append(epi_good_event)
                episode_step_history.append(episode_step)
                
                # save checkpoint model, if a good model is received
                if episode_reward > RECORD_VAL:
                    save_path = cwd +'\\' + str(num_episode) + '_reward' + str(episode_reward) + 'step' + str(episode_step) + '.ckpt'
                    saver.save(RL.sess, save_path)
                break
            total_step += 1
            epi_action_list.append(selected_action)
    # %%draw cost curve
    RL.plot_cost()
    RL.plot_reward(reward_history, 250)
    RL.plot_epiEvent(good_event_history)
    save_path_reward_mat = cwd + '\\' + 'reward_his.mat'
    sio.savemat(save_path_reward_mat, mdict={'reward': reward_history})

else:  
    Dead_states = [[1, 0, 2, 0, 0, 1, 101], [0, 3, 0, 3, 0, 1, 195], [2, 0, 2, 0, 3, 1, 113], 
                   [0, 4, 0, 3, 3, 1, 149]] 
    if Single_model:       
        file_path = r"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\28692_reward610.8500000000016step201.ckpt" # fill in the target ckpt    
        tf.reset_default_graph()  
        S = [0, 0, 0, 0, 0, 0, 0]  
        
        
        S_norm, _ = AGV_norm_state(S)
        
        [generated_states, Problem_state, reach_states] = RL.check_action_AGV_cluster(S, file_path, plant_param, Cluster, K, Delta, Dead_states)
        print(f"checked state number: {len(generated_states)}, problem state number: {len(Problem_state)}")
        print(f"Problem states are: {Problem_state}")
        #"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\28692_reward610.8500000000016step201.ckpt"
        
        
        #Problem states are: [[2, 0, 2, 0, 3, 1, 113](state number in Matlab:2253), 
        #[1, 0, 2, 0, 0, 1, 101](279), 
        #[0, 4, 0, 3, 3, 1, 149](4008), [0, 3, 0, 3, 0, 1, 195](1495)]
        
        #Special_state_set=[[0, 0, 2, 0, 2, 0, 65](1956),
        #[0, 0, 2, 0, 3, 0, 113](2050), [0, 0, 2, 0, 0, 0, 65](240)
        #[0, 0, 0, 3, 2, 0, 205](3619), [0, 0, 0, 3, 3, 0, 149](3713), [0, 0, 0, 3, 0, 0, 205](1243)]
               
        #Delta does
        #STEP = 0, Cluster=1: 
        #Delta = 0.7, 4422 reachable states, including 4 deadlock states
        #Delta = 0.65, 4422 reachable states, including 4 deadlock states
        #Delta = 0.6, 4422 reachable states, including 4 deadlock states
        #Delta = 0.55, 4422 reachable states, including 4 deadlock states
        #Delta = 0.53, 4422 reachable states, including 4 deadlock states
        #Delta = 0.5, 4421 reachable states, including 4 deadlock states
        #Delta = 0.4, 4401 reachable states, including 5 deadlock states
        #Delta = 0.3, 4200 reachable states, including 13 deadlock states
        
        #STEP = 1, Cluster=1:
        #Delta = 0.8, 4422 reachable states, including 4 deadlock states    
        #Delta = 0.7, 4422 reachable states, including 4 deadlock states    
        #Delta = 0.6, 4419 reachable states, including 4 deadlock states    
        #Delta = 0.5, 4406 reachable states, including 4 deadlock states
        #Delta = 0.4, 4357 reachable states, including 4 deadlock states
        #Delta = 0.3, 4017 reachable states, including 4 deadlock states
        #Delta = 0.1, 3859 reachable states, including 4 deadlock states
        #Delta = 0, 3813 reachable states, including 4 deadlock states
        
        #STEP = 2, Cluster=1:
        #Delta = 0.8, 4422 reachable states, including 4 deadlock states 
        #Delta = 0.7, 4422 reachable states, including 4 deadlock states   
        #Delta = 0.6, 4151 reachable states, including 15 deadlock states
        #Delta = 0.5, 3840 reachable states, including 15 deadlock states
        #Delta = 0.4, 3665 reachable states, including 15 deadlock states
        #Delta = 0, 3150 reachable states, including 12 deadlock states
        
       
        #Nonblocking
        #state, Nonblocking_states, step, isDone_test = RL.run_AGV(S, file_path, plant_param, Cluster, K, Delta, Dead_states)
        # Num_Nonstate_set.append(len(Nonblocking_states))
        # IS_done.append(isDone_test)
        # print(f"Nonblocking states number: {len(Nonblocking_states)}")
        #2662 nonblicking states
    else:
        #Sub-model decision method
        S = [0, 0, 0, 0, 0, 0, 0]
        S_norm, _ = AGV_norm_state(S)
        file_path_set = r"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\28692_reward610.8500000000016step201.ckpt", #1229 Nonblocking
        [r"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\28724_reward621.7500000000018step201.ckpt", #646 Nonblocking
        r"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\29517_reward579.8000000000015step201.ckpt",  #759 Nonblocking
        r"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\29578_reward607.7000000000015step201.ckpt",  #973 Nonblocking
        r"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\29670_reward606.783333333335step201.ckpt",   #719 Nonblocking
        r"D:\Programs\Python\AGV_DQN_cluster\2025-04-08\AGV\29674_reward595.083333333335step201.ckpt"]   #651 Nonblocking
        generated_states_full, Problem_state, reach_states = RL.Multi_model(S, file_path_set, plant_param)
        #Multi-model 1229 Nonblocking
    
    
    
    
    
    
    
    
    