import torch

def check_pytorch_status():
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Check number of CUDA devices
        cuda_device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {cuda_device_count}")

        # Print CUDA devices
        for i in range(cuda_device_count):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

        # Check if cuDNN is available
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"cuDNN available: {cudnn_available}")

        # Check if Tensor Cores are available
        # tensor_cores_available = torch.cuda.is_tensor_core_available()
        # print(f"Tensor Cores available: {tensor_cores_available}")

    # Check if MKL (Math Kernel Library) is available
    mkl_available = torch.backends.mkl.is_available()
    print(f"MKL available: {mkl_available}")

    # Check if TorchScript is available
    torchscript_available = torch.cuda.is_available() and torch.version.cuda is not None
    print(f"TorchScript available: {torchscript_available}")

    # Check if TorchVision is installed and its version
    try:
        import torchvision
        print(f"TorchVision version: {torchvision.__version__}")
    except ImportError:
        print("TorchVision is not installed.")

if __name__ == "__main__":
    check_pytorch_status()
