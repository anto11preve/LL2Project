import torch

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU.")
    import torch
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

else:
    print("CUDA is not available. You will use CPU.")