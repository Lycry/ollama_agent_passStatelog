import torch
import timeit
torch.empty(2, 3) # empty tensor (uninitialized), shape (2,3)
torch.rand(2, 3) # random tensor, each value taken from [0,1)
torch.randn(2, 3) # random tensor, each value taken from standard normal distribution
torch.zeros(2, 3, dtype=torch.long) # long integer zero tensor
torch.zeros(2, 3, dtype=torch.double) # double float zero tensor
torch.arange(10)


array = [[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]]
torch.tensor(array)
import numpy as np
array = np.array([[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]])
torch.from_numpy(array)

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
print(torch.rand(2, 3, device="mps"))
print(torch.rand(2, 3).to("mps"))

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double)
print(x.mean(dim=0))

# M = torch.rand(1000, 1000)
# print(timeit.timeit(lambda: M.mm(M).mm(M), number=5000))
#
# N = torch.rand(1000, 1000).to("mps")
# print(timeit.timeit(lambda: N.mm(N).mm(N), number=5000))


x = torch.arange(16).view(4, 4)
print(x)
print(x[-2:, :])

y = torch.arange(4).view(2, 2)
print(y, y.shape)
a = torch.squeeze(y, dim=-2)
print(a, a.shape)