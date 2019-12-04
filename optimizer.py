class InsightFaceOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0
        self.lr = 0.1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        # divide the learning rate at 100K,160K iterations
        if self.step_num in [100000, 160000]:
            self.lr = self.lr / 10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def clip_gradient(self, grad_clip):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def adjust_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print("The new learning rate is %f\n" % (self.optimizer.param_groups[0]['lr'],))
