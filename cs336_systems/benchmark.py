import timeit
import torch
import statistics
import pandas as pd
import torch.cuda.nvtx as nvtx
import cs336_basics.model as model_module

from omegaconf import OmegaConf
from einops import rearrange, einsum

from contextlib import nullcontext
from cs336_basics.model import TransformerLM, AdamW, Linear, RoPE
from cs336_basics.utils import cross_entropy, softmax

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    with nvtx.range("computing attention scores"):
        attn = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k**0.5)

    if mask is not None:
        attn = attn.masked_fill(~mask, float("-inf"))

    with nvtx.range("computing softmax"):  
        attn = softmax(attn, dim=-1)
    with nvtx.range("final matmul"):
        output = einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output

class Annotated_MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len=None,
        theta=None,
        device=None,
        dtype=None
    ):
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.num_heads = num_heads
        d_h = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, **kwargs)
        self.k_proj = Linear(d_model, d_model, **kwargs)
        self.v_proj = Linear(d_model, d_model, **kwargs)
        self.o_proj = Linear(d_model, d_model, **kwargs)

        if max_seq_len is not None:
            self.rope = RoPE(d_h, theta, max_seq_len, device=device)

    def forward(self, x:torch.Tensor, token_positions=None, use_rope=True):
        seq_len = x.shape[-2]
        with nvtx.range("MHA"):
            with nvtx.range("QKV project"):
                Q = self.q_proj(x)
                K = self.k_proj(x)
                V = self.v_proj(x)

            Q = rearrange(Q, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
            K = rearrange(K, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
            V = rearrange(V, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)

            if use_rope and token_positions is not None:
                with nvtx.range("ROPE"):
                    Q = self.rope(Q, token_positions)
                    K = self.rope(K, token_positions)
        
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            attn_output = annotated_scaled_dot_product_attention(Q, K, V, mask=mask)
            attn_output = rearrange(attn_output, "... h seq d_h -> ... seq (h d_h)")
            with nvtx.range('output project'):
                output = self.o_proj(attn_output)
        return output

model_module.MultiHeadAttention = Annotated_MultiHeadAttention

def benchmark(cfg):
    device = torch.device("cuda:0")

    inputs = torch.randint(
        0,
        cfg.vocab_size,
        size=(cfg.batch_size, cfg.model.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        cfg.vocab_size,
        size=(cfg.batch_size, cfg.model.context_length),
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
    for _ in range(cfg.warmup_steps):
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Make sure warmup CUDA kernels are finished
    torch.cuda.synchronize()
    profile_memory = cfg.get('profile_memory', False)
    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    times = []
    # -----------------------
    # Benchmark
    # -----------------------

    with nvtx.range("benchmark"):
        for step in range(cfg.steps):
            with nvtx.range(f"iteration_{step}"):
                if cfg.mode == "forward_only":
                    torch.cuda.synchronize()
                    start = timeit.default_timer()

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if cfg.use_mix else nullcontext():
                        with nvtx.range("forward"):
                            outputs = model(inputs)

                    torch.cuda.synchronize()
                    times.append(timeit.default_timer() - start)
                    del outputs

                elif cfg.mode == "forward_backward":
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    
                    outputs = model(inputs)
                    loss = cross_entropy(outputs, targets)
                    loss.backward()

                    torch.cuda.synchronize()
                    times.append(timeit.default_timer() - start)

                elif cfg.mode == "forward_backward_optimizer":
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if cfg.use_mix else nullcontext():
                        with nvtx.range("zero_grad"):
                            optimizer.zero_grad(set_to_none=True)
                        with nvtx.range("forward"):
                            
                                outputs = model(inputs)
                        with nvtx.range("loss"):
                            loss = cross_entropy(outputs, targets)
                        with nvtx.range("backward"):
                            loss.backward()
                        with nvtx.range("step"):
                            optimizer.step()

                        torch.cuda.synchronize()
                        times.append(timeit.default_timer() - start)
                else:
                    raise ValueError(f"Unknown mode: {cfg.mode}")

    if profile_memory:
        snapshot_path = f"./profiles/memory/{cfg.model_size}_{cfg.model.context_length}_{cfg.mode}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to: {snapshot_path}")

    mean = statistics.mean(times) * 1000
    std = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0

    print(f"Model Size: {cfg.model_size}")
    print(f"Context Length: {cfg.model.context_length}")
    print(f"{cfg.mode} Time: {mean:.3f} ± {std:.3f} ms")

if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    benchmark(cfg)