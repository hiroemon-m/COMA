import torch

class Memory:
    def __init__(self, agent_num, action_dim,story_count):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.story_count = story_count
        self.probs = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.edges = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.features = torch.empty([self.story_count,self.agent_num, 2411])
        self.next_edges = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.next_features = torch.empty([self.story_count,self.agent_num, 2411])
        #エージェントの数だけ行数がある
        self.pi = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.reward = torch.empty([self.story_count,agent_num,1])
        #エージェントの数だけフラグがある


#torch.tenosrでtenosrにする
    def get(self):
        probs = self.probs
        self.edges
        self.features

        pi = torch.empty([self.agent_num,self.story_count,self.agent_num])
        for i in range(self.agent_num):

            pi[i]=self.pi[:,i]

        print("test_pi",pi)
        
    
            #全てのエージェントの行動確率の値
      
        #reward = torch.tensor(self.reward)
        #print(reward.shape)
        #reward (1x4)報酬
        #print(reward)
        return probs, self.edges, self.features, pi, self.reward
    
    def test(self,t):
        self.edges
        self.features

        pi = torch.empty([self.agent_num,self.agent_num])
    
        pi=self.pi[t]         
        print("test_pi",pi)
        return  pi

#初期化の関数
    def clear(self):
        self.probs = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.edges = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.features = torch.empty([self.story_count,self.agent_num,2411])
        #NIPS 2411
        #DBLP3854
        #エージェントの数だけ行数がある
        self.pi = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.reward = torch.empty([self.story_count,self.agent_num,1])
        self.next_edges = torch.empty([self.story_count,self.agent_num,self.agent_num])
        self.next_features = torch.empty([self.story_count,self.agent_num, 2411])