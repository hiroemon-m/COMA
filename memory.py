import torch

class Memory:
    def __init__(self, agent_num, action_dim,story_count):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.story_count = story_count
        self.actions = torch.empty([self.story_count,500,500])
        self.observation_edges = torch.empty([self.story_count,500,500])
        self.observation_features = torch.empty([self.story_count,500,3854])
        #エージェントの数だけ行数がある
        self.pi = torch.empty([self.story_count,self.agent_num,500])
        self.reward = torch.empty([self.story_count,500,1])
        #エージェントの数だけフラグがある


#torch.tenosrでtenosrにする
    def get(self):
        actions = self.actions
        self.observation_edges
        self.observation_features

        pi = torch.empty([500,self.story_count,500])
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

        pi = torch.empty([500,500])
    
        pi=self.pi[t]         
        
                #全てのエージェントの行動確率の
           
            #reward (1x4)報酬
            #print(reward)
          
        return  pi

#初期化の関数
    def clear(self):
        self.actions = torch.empty([self.story_count,500,500])
        self.observation_edges = torch.empty([self.story_count,500,500])
        self.observation_features = torch.empty([self.story_count,500,3854])
        #エージェントの数だけ行数がある
        self.pi = torch.empty([self.story_count,self.agent_num,500])
        self.reward = torch.empty([self.story_count,500
                                   ,1])