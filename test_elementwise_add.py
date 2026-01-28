from pathlib import Path
import torch
import ctypes

torch.set_default_device("npu")

M, N = 1024, 1024

a = torch.randn((M, N), dtype=torch.float)
b = torch.randn((M, N), dtype=torch.float)
c = torch.empty((M, N), dtype=torch.float)

SCRIPT_DIR = Path(__file__).resolve().parent
lib = ctypes.CDLL(f"{SCRIPT_DIR}/test_elementwise_add.so")
result = lib.call(
    ctypes.c_void_p(a.data_ptr()),
    ctypes.c_void_p(b.data_ptr()),
    ctypes.c_void_p(c.data_ptr()),
    torch.npu.current_stream()._as_parameter_
)
print(f"result={result}")

torch.npu.synchronize()
print(f"Kernel Output Is: {c}")

ref_c = a + b
print(f"Golden Output Is: {ref_c}")

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")



