import torch
import torch.nn.functional as F

def quantize_dequantize(x, log_scaling_factor=8):
    scale = 1 << log_scaling_factor
    x_int = torch.round(x * scale).to(torch.int32)
    x_back = x_int.float() / scale
    return x_back

def check_silu_quantization(log_scaling_factor=8):
    x = torch.linspace(-5, 5, steps=1000)

    original_output = F.silu(x)
    x_quant = quantize_dequantize(x, log_scaling_factor)
    quantized_output = F.silu(x_quant)

    error = torch.abs(original_output - quantized_output)

    print("Activation Quantization Check")
    print("-----------------------------")
    print("Activation: SiLU / SwiGLU-style")
    print("Scaling factor: 2^", log_scaling_factor)
    print("Max error:", error.max().item())
    print("Mean error:", error.mean().item())

if __name__ == "__main__":
    check_silu_quantization()
