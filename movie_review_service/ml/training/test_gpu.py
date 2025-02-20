import torch


# Simple test to check if GPU is available that can be run independently before running train.py
def test_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        # Test GPU with a simple operation
        print("\nTesting GPU with tensor operations...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("GPU tensor operation completed successfully")

        # Verify tensor is on GPU
        print(f"\nTensor device: {z.device}")


if __name__ == "__main__":
    test_gpu()
