import torch
import numpy as np


# Initialize a tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)

x_rand = torch.rand_like(x_data, dtype = torch.float)
#print(f"Original tensor: \n {x_data} \n")
#print(f"Ones tensor: \n {x_ones} \n")
#print(f"Random tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#print(f"Random tensor: \n {rand_tensor} \n")
#print(f"Ones tensor: \n {ones_tensor} \n")
#print(f"Zeros tensor: \n {zeros_tensor} \n")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.accelerator.is_available():
    print("Entered if")
    tensor = tensor.to("cuda")

print(f"Device tensor is stored on: {tensor.device}")