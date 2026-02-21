import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("Tensor:\n", x)

# Basic operations
print("Addition:\n", x + 2)
print("Matrix multiplication:\n", x @ x)
print("Element-wise multiplication:\n", x * x)

# Move tensor to GPU (if available)
if torch.cpu.is_available():
    x = x.to("cpu")
    print("Tensor on GPU:\n", x)
