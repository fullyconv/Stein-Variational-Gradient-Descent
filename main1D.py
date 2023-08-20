# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import matplotlib.pyplot as plt
import os
import torch.autograd as autograd
import torch.optim as optim
torch.cuda.empty_cache()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import torch.distributions.MultivariateNormal as MVN
import torch.distributions as D
from torch.distributions.utils import _standard_normal, lazy_property


def kernel(samples):
    kernel_result = torch.square(samples - samples.T)
    kernel_result = torch.exp(-kernel_result)
    return kernel_result


def grad(samples):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    grad_log_prob2 = autograd.grad(torch.mean(kernel(samples), dim=1, keepdim=True), samples,
                                   grad_outputs=torch.ones_like(samples))[0].to(device)
    log_prob = gmm.log_prob(samples).to(device)

    grad_log_prob = autograd.grad(log_prob, samples,
                                  grad_outputs=torch.ones_like(samples).to(device))

    # print( grad_log_prob[0])

    return torch.mean(kernel(samples) * grad_log_prob[0], dim=1, keepdim=True) + grad_log_prob2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mix = D.Categorical(torch.ones(5, ).to(device))
comp = D.Normal(torch.randn(5, ).to(device), torch.rand(5, ).to(device))
gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
samples = gmm.sample(sample_shape=torch.Size([1000, ]))

plt.figure()
plt.title("Target")
plt.hist(samples.cpu().numpy())
plt.show()

normal_d = D.Normal(torch.tensor([0.0]).to(device),
                    torch.tensor([1.0]).to(device))

samples = normal_d.sample(sample_shape=torch.Size([1000])).requires_grad_().to(device)
gmm.log_prob(samples)

epsilon = 0.01
for i in range(1000):
    samples = samples + epsilon * grad(samples)

plt.figure()
plt.title("Samples")
plt.hist(samples.cpu().detach().numpy())
plt.show()
print((samples))

# n_iter=1000
# for i in range(n_iter):
#     torch.mean()

# def mog(n_sample,alpha,sigma,mu):
#     for i in range(i):
#
#     return 0

# def kernel(x,x_hat):
#     return torch.exp(-torch.tensor(x-x_hat)**2)

# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print("kernel(3,5)")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
