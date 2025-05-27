import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check how many GPUs are available
print("Number of GPUs:", torch.cuda.device_count())

# Get the name of the current GPU
if torch.cuda.is_available():
    print("Current GPU Name:", torch.cuda.get_device_name(0))
