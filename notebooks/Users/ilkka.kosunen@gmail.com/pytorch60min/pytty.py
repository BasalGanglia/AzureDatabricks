# Databricks notebook source
from __future__ import print_function
import torch

# COMMAND ----------

x = torch.empty(5,3)
print(x)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testing markup.
# MAGIC also, creating some empty tensors

# COMMAND ----------

x = x.new_ones(5,3, dtype=torch.double)
print(x)

# COMMAND ----------

# Now creating similar tensor with bit of noise
x = torch.randn_like(x, dtype=torch.float)
print(x)

# COMMAND ----------

# MAGIC %md there are several syntaxes for operations
# MAGIC first create another (5,3) tensor:

# COMMAND ----------

y = torch.rand(5,3)

# COMMAND ----------

# first way to add
print(x+y)

# COMMAND ----------

# also so:
print(torch.add(x,y))

# COMMAND ----------

# we can also save the result as so:
result = torch.empty(5,3) # just empty tensor
torch.add(x,y,out=result)
print(result)


# COMMAND ----------

# and even like this
y.add_(x)
print(y)

# COMMAND ----------

# MAGIC %md
# MAGIC # Any operation that mutates a tensor in-place is post-fixed with an _

# COMMAND ----------

# pytorch supports standard np indexing:
print(x[:,1])

# COMMAND ----------

# MAGIC %md
# MAGIC # Resize tensors with torch.view

# COMMAND ----------

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Torch Tensor and NumPy array share their underlying memory locations

# COMMAND ----------

a=torch.ones(5)
print(a)

# COMMAND ----------

b = a.numpy()
print(b)

# COMMAND ----------

# adding to a in place
a.add_(1)
print(a)
print(b)

# COMMAND ----------

# MAGIC %md
# MAGIC #CUDA Tensors
# MAGIC Tensors can be moved to device using .to

# COMMAND ----------

if torch.cuda.is_available():
  device = torch.device("cuda")
  y = torch.ones_like(x,device=device)
  x = x.to(device)
  z = x + y
  print(z)
  print(z.to("cpu",torch.double))

# COMMAND ----------

# MAGIC %md
# MAGIC #AUTOGRAD

# COMMAND ----------

x = torch.ones(2,2,requires_grad=True)
print(x)

# COMMAND ----------

y = x+2
print(y)

# COMMAND ----------

# MAGIC %md 
# MAGIC y was created as a result of an operation, so it has a grad_fn

# COMMAND ----------

z = y*y * 3
out = z.mean()

# COMMAND ----------

print(out)

# COMMAND ----------

print(x.grad)

# COMMAND ----------

out.backward()

# COMMAND ----------

print(x.grad)

# COMMAND ----------

# MAGIC %md
# MAGIC # Neural networks
# MAGIC nns can be constructed using the torch.nn package.
# MAGIC 
# MAGIC - nn depends on **autograd* to define models and differentiate them. 
# MAGIC - An *nn.Module* contains layers, and a method *forward(input)* that returns the output.