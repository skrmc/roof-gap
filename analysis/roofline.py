"""Hand-derived per-op roofline for one decode step of a dense GQA transformer (Llama-style).

Every op is (name, FLOPs, bytes moved to/from HBM). Step time under the roofline is
sum_i max(F_i / peak_flops, B_i / peak_bw), i.e. ops are serialized and each is bound by one ceiling.
"""
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    d: int          # hidden size
    L: int          # layers
    hq: int         # query heads
    hkv: int        # KV heads
    dh: int         # head dim
    ffn: int        # MLP intermediate size
    vocab: int
    wbytes: int = 2   # weight bytes (bf16)
    kvbytes: int = 2  # KV cache bytes (bf16)

    @property
    def kv_per_token(self):  # bytes of K and V for one token across all layers
        return 2 * self.L * self.hkv * self.dh * self.kvbytes

    def linears(self):  # (name, N_out, K_in) per layer, vLLM-style fused QKV and gate/up
        return [("qkv", (self.hq + 2 * self.hkv) * self.dh, self.d), ("o", self.d, self.hq * self.dh),
                ("gate_up", 2 * self.ffn, self.d), ("down", self.d, self.ffn)]

    @property
    def params(self):
        per_layer = sum(n * k for _, n, k in self.linears()) + 2 * self.d
        return self.L * per_layer + 2 * self.vocab * self.d + self.d  # + embed + lm_head + final norm


@dataclass
class HW:
    name: str
    flops: float  # dense BF16 FLOP/s
    bw: float     # HBM B/s


A100_SPEC = HW("A100-SXM4-40GB datasheet", 312e12, 1555e9)
LLAMA31_8B = Model("Llama-3.1-8B", d=4096, L=32, hq=32, hkv=8, dh=128, ffn=14336, vocab=128256)


def decode_ops(m: Model, b: int, s: int):
    """Ops of one decode step: b sequences, each attending over s cached tokens (+ the new one)."""
    a, ops = m.wbytes, []
    for name, n, k in m.linears():
        ops.append((name, m.L * 2 * b * n * k, m.L * a * (n * k + b * k + b * n)))
    ctx = s + 1
    # attention: QK^T and PV, 2*dh FLOPs per (head, key) each; reads the whole K and V cache once
    ops.append(("attn", m.L * 4 * b * m.hq * m.dh * ctx,
                m.L * (b * ctx * 2 * m.hkv * m.dh * m.kvbytes + 2 * b * m.hq * m.dh * a)))
    ops.append(("kv_write", 0, b * m.kv_per_token))
    # elementwise: 2x RMSNorm (r+w b*d), RoPE on q,k, SiLU*mul (r 2*ffn, w ffn), 2x residual add (r 2, w 1)
    ew = b * a * (2 * 2 * m.d + 2 * (m.hq + m.hkv) * m.dh + 3 * m.ffn + 2 * 3 * m.d)
    ops.append(("elementwise", m.L * b * (10 * m.d + 3 * m.ffn), m.L * ew))
    ops.append(("lm_head", 2 * b * m.vocab * m.d, a * (m.vocab * m.d + b * m.d + b * m.vocab)))
    return ops


def step_time(m: Model, hw: HW, b: int, s: int):
    return sum(max(f / hw.flops, B / hw.bw) for _, f, B in decode_ops(m, b, s))


if __name__ == "__main__":
    m, hw = LLAMA31_8B, A100_SPEC
    print(f"params={m.params/1e9:.3f}B  kv/token={m.kv_per_token} B  ridge={hw.flops/hw.bw:.0f} FLOP/B")
    w = sum(B for n, _, B in decode_ops(m, 1, 0) if n in ("qkv", "o", "gate_up", "down", "lm_head"))
    print(f"weight bytes read per step ~ {w/1e9:.2f} GB  ->  floor {w/hw.bw*1e3:.2f} ms/step")
    print(f"b*s where KV bytes == weight bytes: {w/m.kv_per_token:,.0f} tokens")
    print("\n   b      s   T_step(ms)  tok/s     F(TFLOP)  B(GB)  AI    weights%  kv%")
    for b in (1, 8, 32, 64, 128, 256):
        for s in (512, 4096):
            if b * s > 160_000:
                continue
            ops = decode_ops(m, b, s)
            F, B = sum(o[1] for o in ops), sum(o[2] for o in ops)
            t = step_time(m, hw, b, s)
            kv = next(o[2] for o in ops if o[0] == "attn")
            print(f"{b:4d} {s:6d} {t*1e3:10.2f} {b/t:8.0f} {F/1e12:10.3f} {B/1e9:6.2f} {F/B:5.1f} "
                  f"{w/B*100:8.0f}% {kv/B*100:4.0f}%")
