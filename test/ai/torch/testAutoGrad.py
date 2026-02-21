import torch
from torchviz import make_dot
# Define a tensor with gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y + 3
print("Gradient at x=2:", x.grad)
graph = make_dot(y, params={"x": x})
graph.render("computational_graph", format="png")
