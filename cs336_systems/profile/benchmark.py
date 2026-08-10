import timeit
import torch
import statistics
import pandas as pd
from omegaconf import OmegaConf

from cs336_basics.model import TransformerLM, AdamW
from cs336_basics.utils import cross_entropy


def benchmark(cfg):
    model_size = "small"
    batch_size = 4
    vocab_size = 10000
    device = torch.device("cuda:0")

    inputs = torch.randint(
        0,
        vocab_size,
        size=(batch_size, cfg.model.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        vocab_size,
        size=(batch_size, cfg.model.context_length),
        device=device,
        dtype=torch.long,
    )

    model = TransformerLM(
        vocab_size=10000,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        d_ff=cfg.model.d_ff,
        rope_theta=10000
    ).to(device)

    optimizer = AdamW(model.parameters())
    model.train()

    # Warmup
    warmup_steps = 5
    for _ in range(warmup_steps):
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Make sure warmup CUDA kernels are finished
    torch.cuda.synchronize()
    eval_steps = 10

    forward_times = []
    backward_times = []
    optimizer_times = []

    # -----------------------
    # Benchmark
    # -----------------------
    for _ in range(eval_steps):

        optimizer.zero_grad(set_to_none=True)

        # ===== Forward =====
        torch.cuda.synchronize()
        start = timeit.default_timer()

        outputs = model(inputs)

        torch.cuda.synchronize()
        forward_times.append(timeit.default_timer() - start)

        # ===== Backward =====
        torch.cuda.synchronize()
        start = timeit.default_timer()

        loss = cross_entropy(outputs, targets)
        loss.backward()

        torch.cuda.synchronize()
        backward_times.append(timeit.default_timer() - start)

        # ===== Optimizer step =====
        torch.cuda.synchronize()
        start = timeit.default_timer()

        optimizer.step()

        torch.cuda.synchronize()
        optimizer_times.append(timeit.default_timer() - start)

    forward_mean = statistics.mean(forward_times) * 1000
    forward_std = statistics.stdev(forward_times) * 1000

    backward_mean = statistics.mean(backward_times) * 1000
    backward_std = statistics.stdev(backward_times) * 1000

    optimizer_mean = statistics.mean(optimizer_times) * 1000
    optimizer_std = statistics.stdev(optimizer_times) * 1000

    print(f"Forward:   {forward_mean:.3f} ± {forward_std:.3f} ms")
    print(f"Backward:  {backward_mean:.3f} ± {backward_std:.3f} ms")
    print(f"Optimizer: {optimizer_mean:.3f} ± {optimizer_std:.3f} ms")

if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    benchmark(cfg)