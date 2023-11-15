import torch

class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = torch.empty([5,32,32])
        self.observation_edges = torch.empty([5,32,32])
        self.observation_features = torch.empty([5,32,2411])
        #エージェントの数だけ行数がある
        self.pi = torch.empty([5,self.agent_num,32])
        self.reward = torch.empty([5,32,1])
        #エージェントの数だけフラグがある


#torch.tenosrでtenosrにする
    def get(self):
        actions = self.actions
        self.observation_edges
        self.observation_features

        pi = torch.empty([32,5,32])
        for i in range(self.agent_num):

            pi[i]=self.pi[:,i]
    
            #全てのエージェントの行動確率の値
      
        #reward = torch.tensor(self.reward)
        #print(reward.shape)
        #reward (1x4)報酬
        #print(reward)
        return actions, self.observation_edges, self.observation_features, pi, self.reward
    
    def test(self,t):
        actions = self.actions
        self.observation_edges
        self.observation_features

        pi = torch.empty([32,32])
    
        pi=self.pi[t]         
        
                #全てのエージェントの行動確率の値
        
           
            #reward (1x4)報酬
            #print(reward)
          
        return  pi

#初期化の関数
    def clear(self):
        self.actions = torch.empty([5,32,32])
        self.observation_edges = torch.empty([5,32,32])
        self.observation_features = torch.empty([5,32,2411])
        #エージェントの数だけ行数がある
        self.pi = torch.empty([5,self.agent_num,32])
        self.reward = torch.empty([5,32,1])