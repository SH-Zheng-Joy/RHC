import torch
import torch.autograd as autograd
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from torch.distributions import Normal
# import torch.optim as optim
import collections
# 将梯度置零
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)
            
# 计算雅可比矩阵
def compute_jacobian(inputs, output, create_graph=False):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())  # Class X Batch X Size
    grad_output = torch.zeros(*output.size()) # Batch X Classes
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        jacobian[i] = inputs.grad.clone()

    return torch.transpose(jacobian, dim0=0, dim1=1)

class DetPolicyWrapper(object):

    def __init__(self, model, policy, T=10, lr=0.1, eps=1e-1, reg=1.0):
        self.model = model
        self.policy = policy
        self.T = T
        self.eps = eps
        self.reg = reg

        self.state_dim = model.num_states
        self.action_dim = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

        self.u = torch.zeros(T, self.action_dim).to(self.device)
        self.u.requires_grad = True

#         self.lr = lr # alp commented out 6/10
#         self.optim = optim.SGD([self.u], lr=lr) # alp commented out 6/10

    def reset(self):
        with torch.no_grad():
            self.u.zero_()

    def get_mode_insertion(self, state):
        #前向滚动预测
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            # `unsqueeze` 方法用于在指定的维度上扩展张量的维度。具体来说，`unsqueeze(0)` 会在张量的第 0 维（即最外层）添加一个新的维度。
            x = []
            for u in self.u:
                x.append(s.clone())
                u_t, log_std_t = self.policy(s)
                # pi = Normal(torch.tanh(u_t), log_std_t.exp()) # unused (commmented out alp 5/10)
                u_app = torch.tanh(u_t+u.unsqueeze(0)) #+ torch.randn_like(u_t) * torch.exp(log_std_t)
                s, r = self.model.step(s, u_app)

        # compute those derivatives
        x = torch.cat(x) # 这里将x中的元素拼接成在一起
        x.requires_grad = True

        # self.optim.zero_grad() #unused (commmented out alp 5/10)
        u_p, log_std_p = self.policy(x)

        pred_state, pred_rew = self.model.step(x, torch.tanh(self.u+u_p)) # torch.tanh added here (alp 5/10)

        dfdx = compute_jacobian(x, pred_state) #状态转移函数f相对于状态s的偏导数，注意这里将T个批次的偏导全部计算了
        dfdu = compute_jacobian(self.u, pred_state) #状态转移函数f相对于动作a的偏导数
        dldx = compute_jacobian(x, pred_rew) #奖励函数r相对于状态s的偏导数
        
        # 计算协变量rho,并返回最优动作和对应的协变量
        with torch.no_grad():
            rho = torch.zeros(1, self.state_dim).to(self.device)
            for t in reversed(range(self.T)):
                rho = dldx[t] + rho.mm(dfdx[t])
            rho = torch.clamp(rho, -1,+1)
            sig = torch.pow(log_std_p[0].exp().unsqueeze(0),2)
            _u = (sig * rho.mm(dfdu[0])).unsqueeze(0)
            u1 = torch.clamp(torch.tanh(u_p[0].unsqueeze(0)), -1, 1)
            u2 = torch.clamp(torch.tanh(u_p[0])+_u[0],-1,1)
            f1, _ = self.model.step(x[0].unsqueeze(0), u1)
            f2, _ = self.model.step(x[0].unsqueeze(0), u2)
            self.u.grad.zero_()
            return (torch.clamp(_u[0]+u_p[0], -1, +1).squeeze(0).cpu().clone().numpy(),
                    rho.mm((f2-f1).T).squeeze().cpu().clone().numpy())

    def __call__(self, state, epochs=1):

#         for epoch in range(epochs):
#             s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             cost = 0.
#             # t = 0 # unused (commmented out alp 5/10)
#             for u in self.u:
#                 u_t, log_std_t = self.policy(s)
#                 pi = Normal(torch.tanh(u_t), log_std_t.exp())
#                 u_app = torch.tanh(u_t+u.unsqueeze(0)) #+ torch.randn_like(u_t) * torch.exp(log_std_t)
#                 s, r = self.model.step(s, u_app)
#                 cost = cost - (r + pi.log_prob(u_app).mean())
#                 # t += 1 # unused (commmented out alp 5/10)
#             self.optim.zero_grad()
#             cost.backward()
#             self.optim.step()

        with torch.no_grad():
#             s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             u_t, log_std_t = self.policy(s)
#             v = u_t + torch.randn_like(log_std_t) * torch.exp(log_std_t)
#             u = torch.tanh(v.squeeze() + self.u[0]).cpu().clone().numpy()
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
        u, rho = self.get_mode_insertion(state)
        with torch.no_grad():
            return u, rho
