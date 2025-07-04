#import gymnasium as gym
import gym
import register  # Ortamı kayıt ediyoruz
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import datetime

# Model tanımlanıyor.
# DQN sınıfı, Deep Q-Network (DQN) algoritmasında kullanılan sinir ağı modelini tanımlar.
# nn.Module PyTorch'ta tüm modellerin türetildiği temel sınıftır. Bu sınıftan türetip model yazılıyor.

#Girdi olarak bir state alır.
#Gizli katmanda ReLU aktivasyonla işler.
#Çıkış olarak her aksiyon için tahmini Q-değeri döndürür.

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        #in_states: Giriş boyutu. Yani bir state'in kaç boyutlu olduğunu belirtir.
        #h1_nodes: İlk gizli katmandaki nöron sayısı (örneğin 64 veya 128 gibi).
        #out_actions: Ortamda yapılabilecek toplam aksiyon sayısı. (Her bir aksiyon için bir Q-değeri üretilecek.)
        
        # Network katmanları tanımlanıyor.
        # Giriş olarak in_states boyutlu bir vektör alır, h1_nodes boyutlu bir çıkış üretir.
        self.fc1 = nn.Linear(in_states, h1_nodes)   # İlk fully connected layer
        
        # Gizli katmandan gelen verileri alıp, her bir aksiyon için bir Q-değeri üretir.
        self.out = nn.Linear(h1_nodes, out_actions) # output layer w

    # x: Ağa verilen giriş state vektörü
    def forward(self, x):
        # Girdi fc1 katmanından geçer, ardından ReLU aktivasyon fonksiyonu uygulanır.
        # Negatif değerleri sıfırlar.
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        
        # Gizli katmandan çıkan veri out katmanına girer ve her aksiyon için Q-değeri üretilir.
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
# Ajanın deneyimlerini (transition’ları) saklayan ve eğitimde rastgele örneklemeye 
# izin veren bir deneyim havuzu (experience replay buffer).
#Ajan oynadıkça "geçmiş tecrübelerini" bu sınıfa kaydeder.
#Eğitim yaparken bu eski tecrübelerden rastgele küçük parçalar (mini-batch) alınarak model güncellenir.
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
        #maxlen belleğin kapasitesi kaç kayıt tutulacak
        #kapasite dolarsa en eski kayıt silinir yenisi yazılır (FIFO)
    
    #ajanın her step sonucu (transition) belleğe eklenir.
    #transition (state,action,reward,next_state,done) değerlerini içerir.
    def append(self, transition):
        self.memory.append(transition)

    # Hafızadan rastgele sample_size kadar örnek çeker.
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    # Bellekte şu an kaç deneyim olduğunu verir.
    def __len__(self):
        return len(self.memory)


# Deep Q-Learning
class VardiyaDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # policy and target network kaç adımda bir güncellenecek. Her güncellemede policy target'a kopyalanır.
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # Replay memory'den eğitim için rastgele seçilecek örnek sayısı.

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Sonra tanımlanacak. Ağırlıkların güncellenmesini sağlayacak.

    #ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)
    ACTIONS = ['S','I','O']     # for printing 0,1,2 => SagaGit,Işaretle,OncekiVardiya

    
    # Train the Vardiya environment
    
    #Ortamı oluşturur
    #Ajanı eğitir (epsilon-greedy ile aksiyon alarak)
    #Replay memory'e deneyim ekler
    #Mini-batch’lerle ağı optimize eder
    #Target ağı periyodik olarak günceller
    #Eğitim sonrası sonuçları kaydeder ve görselleştirir
    
    def train(self, episodes, render=False):
        
        env = gym.make('VardiyaEnv-v0')      # Vardiya ortamı oluşturuluyor.
        
        num_states = env.observation_space.n # Toplam state sayısı
        num_actions = env.action_space.n     # Toplam aksiyon sayısı
        
        epsilon = 1                          # 1 olması 100% random actions demek
        memory = ReplayMemory(self.replay_memory_size) # Replay memory oluşturuluyor.

        # Policy and target network oluşturuluyor. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())


        # Policy network optimize ediliyor. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        # Episode sayısı kadar oynanacak
        for i in range(episodes):
            #Oyunu yeniden başlatıyor.
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Ajan oyun terminated olana kadar oynar.
            while(not terminated): #and not truncated):

                # Action seçimi başlarda epsilon-greedy, eğitildikçe deneyimleri arasından
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Action çalıştırılır
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Elde edilen deneyim hafızaya yazılır.
                memory.append((state, action, new_state, reward, terminated)) 

                # Sonraki state geçilir.
                state = new_state

                # Step sayacı bir artırılır.
                step_count+=1

            # Her bir episode için ödül sıfırdan büyükse toplanır.
            if reward > 0:
                rewards_per_episode[i] = reward

            # Ödül almış en az bir episode olmuşsa ve bellekte mini_batch_sizedan fazla veri var ise
            # mini_batch ile öğrenme yapılıyor.
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Epsilon azaltılıyor. Giderek deneyim ön plana çıkacak.
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Policyden networke kopyalama yapılıyor.
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Episodelar bitti ortam kapatılıyor.
        env.close()

        # Eğitilen ağ (policy network) kaydediliyor.
        torch.save(policy_dqn.state_dict(), "vardiya.pt")

        # Eğitim süreci grafik haline getiriliyor.
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Sonuç png dosyası olarak kaydediliyor.
        plt.savefig('vardiya.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Giriş katmanının kaç nöron aldığını (yani state vektörünün uzunluğu) öğreniyoruz.
        num_states = policy_dqn.fc1.in_features

        #mini_batch örneği için policyden gelen q tahmini ve hedef q değeri için listeler
        current_q_list = []
        target_q_list = []

        # her bir transition için
        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # oyun bitti ise hedef mevcut ödül olacak
                target = torch.FloatTensor([reward])
            else:
                # oyun bitmemiş ise Bellman denklemine göre hesaplanıyor.
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Policy ağındaki mevcut state için q değerleri alınıp listeye ekleniyor.
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Target ağındaki q değerleri alınıyor.
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Alınan action için target değeri güncelleniyor. Diğer actionlar aynı kalıyor.
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Baştaki cuurent_q_list ve target_q_list değerlerine göre kayıp değeri hesaplanıyor.
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad() # Gradientleri sıfırla
        loss.backward()            # Loss için geri yayılım yap
        self.optimizer.step()      # Optimizasyon adımı uygula (ağırlıkları güncelle)

    
    #Integer değerli state => tensor gösterimine çevriliyor.
    #Parameters: state=1, num_states=225
    #Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Eğitim sonuçları ile test
    def test(self, episodes):
        # Create Vardiya instance
        #env = gym.make('Taxi-v3', render_mode='human')
        env = gym.make('VardiyaEnv-v0', render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("vardiya.pt"))
        policy_dqn.eval()    # switch model to evaluation mode


        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()



if __name__ == '__main__':

    vardiya = VardiyaDQL()
    
    start_time = datetime.datetime.now()
    print(f"Başlangıç Zamanı: {start_time.strftime('%H:%M:%S')}")
    
    #vardiya.train(500)
    vardiya.test(100)
    
    end_time = datetime.datetime.now()
    print(f"Bitiş Zamanı: {end_time.strftime('%H:%M:%S')}")
    
    
