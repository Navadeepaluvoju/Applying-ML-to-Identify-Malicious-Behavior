import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import torch

# TensorFlow Conversion
def convert_tf_model(model_path, test_data):
    # Load the GPU-trained model
    model = load_model(model_path)

    # Set TensorFlow to run on the CPU
    tf.config.set_visible_devices([], 'GPU')

    # Evaluate the model on the CPU
    accuracy = model.evaluate(test_data, verbose=0)
    print(f"Accuracy on CPU: {accuracy}")

    # Save the CPU-compatible model
    model.save("ocr_model_cpu.h5")

# PyTorch Conversion
def convert_pytorch_model(model_path, inputs, labels):
    # Load the GPU-trained model
    model = torch.load(model_path)

    # Transfer the model to the CPU
    model = model.to('cpu')
    model.eval()

    # Evaluate the model on the CPU
    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        accuracy = (predictions == labels).sum().item() / labels.size(0)
        print(f"Accuracy on CPU: {accuracy}")

    # Save the CPU-compatible model
    torch.save(model, "ocr_model_cpu.pth")

# Performance Evaluation
def measure_fps(model, data_loader, device):
    model.to(device)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    end_time = time.time()

    fps = len(data_loader) / (end_time - start_time)
    return fps

# Example usage (replace with your actual data)
if __name__ == "__main__":
    tf_model_path = "ocr_model_gpu.h5"
    tf_test_data = ... 
    convert_tf_model(tf_model_path, tf_test_data)

    pytorch_model_path = "ocr_model_gpu.pth"
    pytorch_inputs, pytorch_labels = ... 
    convert_pytorch_model(pytorch_model_path, pytorch_inputs, pytorch_labels)