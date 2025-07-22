# myenv/test_env.py

import gym
import register  # Ortamı kayıt ediyoruz
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import datetime
#env = gym.make('DroneEnv-v0')

def save_q_table_array(q_table, filename="q_table.txt"):
    with open(filename, "w") as f:
        for state_idx, actions in enumerate(q_table):
            line = f"{state_idx}:{actions.tolist()}\n"
            f.write(line)

def load_q_table_array(filename="q_table.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()

    q_table = []
    for line in lines:
        state_str, actions_str = line.strip().split(":")
        actions = eval(actions_str)
        q_table.append(actions)

    return np.array(q_table)

def run(episodes, is_training, render=False):

    #env = gym.make('Taxi-v3', render_mode='human' if render else None)
    if (is_training==False):
        env = gym.make('VardiyaEnv-v0', render_mode='human')
    else:
        env = gym.make('VardiyaEnv-v0')

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 500 x 6 array
    else:
        q = load_q_table_array()
        '''
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()
        '''
    learning_rate_a = 0.9 # alpha or learning rate 0 hiçbir şey öğrenmez, 1 son deneyim öğrenilir önceki bilgi silinir. 
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    # 0 anlık ödülü önemser, 0.9 gelecek ödül önemli uzun vadeli düşünür.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    #yavaş yavaş random bırakılıyor
    
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        #print(i)
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        rewards = 0
        while(not terminated and not truncated):
            #print(terminated)
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=down,1=up,2=right,3=left,4=pickup,5=dropoff 
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            rewards += reward

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001


        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')
    #print(q)

    if is_training:
        save_q_table_array(q)

    '''
    if is_training:
        f = open("taxi.pkl","wb")
        pickle.dump(q, f)
        f.close()
    '''
    
if __name__ == '__main__':
    #run(15)
    start_time = datetime.datetime.now()
    print(f"Başlangıç Zamanı: {start_time.strftime('%H:%M:%S')}")
    
    run(100, is_training=False, render=True)
    #run(15000, is_training=True, render=False)
    
    end_time = datetime.datetime.now()
    print(f"Bitiş Zamanı: {end_time.strftime('%H:%M:%S')}")
