import torch

class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = []
        self.observation_edges = []
        self.observation_features = []
        #エージェントの数だけ行数がある
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        #エージェントの数だけフラグがある


#torch.tenosrでtenosrにする
    def get(self):
        actions = self.actions
        self.observation_edges
        self.observation_features

        pi = []
        for i in range(self.agent_num):
            #self.pi[i]:エージェントiの方策の配列
            #self.pi[i]を列方向に結合し、
            #print(self.pi[i])
            #print(len(self.pi[i]))
            pi.append(self.pi[i])
    
            #全てのエージェントの行動確率の値
      
        reward = torch.tensor(self.reward)
        #reward (1x4)報酬
        #print(reward)
        return actions, self.observation_edges, self.observation_features, pi, reward
    
    def test(self):
            

            pi = []
            for i in range(self.agent_num):
                #self.pi[i]:エージェントiの方策の配列
                #self.pi[i]を列方向に結合し、
                #print(self.pi[i])
                #print(len(self.pi[i]))
                pi.append(self.pi[i])
        
                #全てのエージェントの行動確率の値
        
           
            #reward (1x4)報酬
            #print(reward)
          
            return  pi

#初期化の関数
    def clear(self):
        self.actions = []
        self.observation_features = []
        self.observation_edges = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []