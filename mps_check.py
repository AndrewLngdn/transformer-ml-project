import time, torch

print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# tiny matmul to tickle the device
x = torch.randn(2048, 2048, device=device)
t0 = time.time()
y = x @ x
if device == "mps":
    torch.mps.synchronize()  # wait for GPU
dt = time.time() - t0
print("ok:", float(y.mean()), "| elapsed:", round(dt, 4), "s")