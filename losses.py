import torch 
from torch.nn import functional as F


class BootstrappedCE(torch.nn.Module):
    """from https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch"""
    def __init__(self, start_warm=0, end_warm=1, top_p=0.25):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            #return F.cross_entropy(input, target), 1.0
            return F.cross_entropy(input, target)

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        #return loss.mean(), this_p
        return loss.mean()

class LinearIncreaseScheduler:
    def __init__(self, 
                start_weight = 0.0,
                max_weight = 0.5, 
                num_steps = 2000):
        self.orig_weight = start_weight
        self.weight = start_weight
        assert((start_weight + max_weight) < 1.0) 
        if num_steps == 0:
            num_steps = 1
        self.factor = (1 - start_weight - max_weight)/num_steps 

    def update(self, it):
        self.weight = self.orig_weight +  it * self.factor
        #self.weight = min(max(self.weight, 0), 1) 
        return self.weight 

class LinearDecreaseScheduler:
    def __init__(self,
                 start_weight = 0.15,
                 max_weight = 0.01,
                 num_steps = 2000): 
        self.orig_weight = start_weight
        self.weight = start_weight
        self.max_weight = max_weight
            
        if num_steps == 0:
            num_steps = 1
        self.factor = (start_weight - max_weight)/num_steps 

    def update(self, it):
        self.weight = self.weight - it * self.factor
        self.weight = min(max(self.weight, self.max_weight), self.orig_weight) 
        print(f"weight {self.weight}") 
        return self.weight 


class ScheduledWeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, 
                 start_weight = 0.01,
                 max_weight = 0.5, 
                 scheduler = LinearDecreaseScheduler,
                 num_steps = 2000): 
        super(ScheduledWeightedCrossEntropyLoss, self).__init__() 
        self.scheduler = scheduler(start_weight = start_weight,
                                   max_weight = max_weight,
                                   num_steps = num_steps) 

    def forward(self, input, target, it): 
        weight = self.scheduler.update(it) 
        weight_tensor = torch.tensor([weight, 1-weight]).to(input.device) 
        return F.cross_entropy(input, target, weight=weight_tensor, reduction='mean')
