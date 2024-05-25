
for i in range(storycount):
     
            old_policy = self.memory.actions[i]
            new_policy,_ =  self.new_actor.forward(self.memory.features[i],self.memory.edges[i])
            ratio =torch.exp(torch.log(new_policy+1e-7) - torch.log(old_policy+1e-7))
            ratio_clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
            G = G_r[i] - baseline[i]
        
            loss_unclipped = ratio * G
            loss_clipped = ratio_clipped * G
            loss = torch.min(loss_unclipped, loss_clipped)
            # 最大化のために-1を掛ける
            loss = -loss.mean()
            #print("loss",loss)
  
            # 勾配の計算と適用
            self.new_actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.new_actor_optimizer.step()
            losses.append(loss)
            for name, param in self.new_actor.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad: {param.grad}")
                else:
                    print(f"{name} grad is None")