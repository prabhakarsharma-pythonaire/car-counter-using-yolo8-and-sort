
import torch

print(torch.cuda.is_available())  # Check if GPU is available
print(torch.cuda.device_count())  # Check number of GPUs available
print(torch.cuda.get_device_name(0))  # Check name of GPU (if available)

# Move tensor to GPU (if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

