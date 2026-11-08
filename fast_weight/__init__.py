# Adaptation of the original code from
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/__init__.py
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
# Modifications Copyright (c) 2021 Kazuki Irie

import os
import torch
from torch.utils.cpp_extension import load
# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'fast_weight_cuda.cu')

causal_dot_product_cuda = None
causal_dot_backward_cuda = None

def _ensure_cuda_ext():
    global causal_dot_product_cuda, causal_dot_backward_cuda
    if causal_dot_product_cuda is None and torch.cuda.is_available():
        ext_cuda = load(
            name="fast_weight_forward", sources=[filename], verbose=True)
        causal_dot_product_cuda = ext_cuda.fast_weight_forward
        causal_dot_backward_cuda = ext_cuda.fast_weight_backward


def fast_weight_delta_pytorch(Q, K, V, beta, W, stateful=False):
    """Exact differentiable PyTorch reference fallback for DeltaNet fast weights."""
    N, H, L, E = Q.shape
    M = V.shape[-1]
    q_seq = Q.permute(2, 0, 1, 3).reshape(L, N * H, E)
    k_seq = K.permute(2, 0, 1, 3).reshape(L, N * H, E)
    v_seq = V.permute(2, 0, 1, 3).reshape(L, N * H, M)
    b_seq = beta.permute(2, 0, 1, 3).reshape(L, N * H, 1)

    # Note: in fast_weight_cuda.cu, W has shape (N, H, E, M) where E is key/query dim and M is value dim.
    W_cur = W.reshape(N * H, E, M)
    out_list = []
    for pos in range(L):
        k_t = k_seq[pos].unsqueeze(1)  # (N*H, 1, E)
        q_t = q_seq[pos].unsqueeze(1)  # (N*H, 1, E)
        v_old = torch.bmm(k_t, W_cur).squeeze(1)  # (N*H, M)
        v_insert = b_seq[pos] * (v_seq[pos] - v_old)  # (N*H, M)
        W_cur = W_cur + torch.bmm(k_t.transpose(1, 2), v_insert.unsqueeze(1))
        out_t = torch.bmm(q_t, W_cur).squeeze(1)  # (N*H, M)
        out_list.append(out_t.reshape(N, H, M))

    out = torch.stack(out_list, dim=2)  # (N, H, L, M)
    if stateful:
        return out, W_cur.reshape(N, H, E, M)
    return out


class DeltaFastWeight(torch.autograd.Function):
    """Fast weight using the delta update rule."""

    @staticmethod
    def forward(ctx, Q, K, V, beta, W):
        _ensure_cuda_ext()
        device = Q.device
        N, H, L, E = Q.shape
        _, _, _, M = V.shape

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        beta = beta.contiguous()
        W_work = W.detach().contiguous().clone()

        out = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)
        V_old = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)
        V_insert = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)

        causal_dot_product_cuda(
            Q.data,
            K.data,
            V.data,
            beta.data,
            V_old,
            V_insert,
            W_work,
            out
        )

        ctx.save_for_backward(Q, K, V, beta, V_old, V_insert, W_work)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        _ensure_cuda_ext()
        Q, K, V, beta, V_old, V_insert, W = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        grad_beta = torch.zeros_like(beta)
        grad_W = torch.zeros_like(W)
        W_copy = W.detach().clone()

        causal_dot_backward_cuda(
            Q.data,
            K.data,
            V.data,
            beta.data,
            V_old.data,
            V_insert.data,
            grad_out,
            W_copy,
            grad_W,
            grad_Q,
            grad_K,
            grad_V,
            grad_beta
        )

        return grad_Q, grad_K, grad_V, grad_beta, grad_W


class StatefulDeltaFastWeight(torch.autograd.Function):
    """Fast weight using the delta update rule."""

    @staticmethod
    def forward(ctx, Q, K, V, beta, W):
        _ensure_cuda_ext()
        device = Q.device
        N, H, L, E = Q.shape
        _, _, _, M = V.shape

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        beta = beta.contiguous()
        W_out = W.detach().contiguous().clone()

        out = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)
        V_old = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)
        V_insert = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)

        causal_dot_product_cuda(
            Q.data,
            K.data,
            V.data,
            beta.data,
            V_old,
            V_insert,
            W_out,
            out
        )

        ctx.save_for_backward(Q, K, V, beta, V_old, V_insert, W_out.detach().clone())

        return out, W_out

    @staticmethod
    def backward(ctx, grad_out, grad_W):
        _ensure_cuda_ext()
        Q, K, V, beta, V_old, V_insert, W = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        grad_beta = torch.zeros_like(beta)
        grad_W_work = grad_W.contiguous().clone() if grad_W is not None else torch.zeros_like(W)
        W_copy = W.detach().clone()

        causal_dot_backward_cuda(
            Q.data,
            K.data,
            V.data,
            beta.data,
            V_old.data,
            V_insert.data,
            grad_out,
            W_copy,
            grad_W_work,
            grad_Q,
            grad_K,
            grad_V,
            grad_beta
        )

        return grad_Q, grad_K, grad_V, grad_beta, grad_W_work


def fast_weight_delta(Q, K, V, beta, W):
    if Q.is_cuda:
        return DeltaFastWeight.apply(Q, K, V, beta, W)
    return fast_weight_delta_pytorch(Q, K, V, beta, W, stateful=False)


def stateful_fast_weight_delta(Q, K, V, beta, W):
    if Q.is_cuda:
        return StatefulDeltaFastWeight.apply(Q, K, V, beta, W)
    return fast_weight_delta_pytorch(Q, K, V, beta, W, stateful=True)

if __name__ == '__main__':
    import torch
    torch.manual_seed(111)
    # Tests pass if the relative difference compared with
    # the corresponding torch autograd computation
    # is smaller than a threshold.

    # Ideally should be tested with double...
    rel_threshold = 1e-3

    # from https://github.com/idiap/fast-transformers/blob/master/tests/causal_product/test_causal_product_gpu.py
    def max_relative_error(a, b, eps=1e-6):
        return torch.abs((b - a) / (torch.abs(b) + eps)).max().item()

    print('##########################')
    print('# Test forward pass')
    print('##########################')

    bsz, n_head, slen, d_head = 3, 5, 11, 64
    v_dim = 64

    # (B, H, len, dim)
    q0 = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    k0 = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    v0 = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    beta0 = torch.sigmoid(torch.rand(bsz, n_head, slen, 1, device='cuda'))

    q0 = q0 / q0.sum(dim=-1, keepdim=True)
    k0 = k0 / k0.sum(dim=-1, keepdim=True)

    q1 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    k1 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v1 = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    beta1 = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    q1.data = q0.data
    k1.data = k0.data
    v1.data = v0.data
    beta1.data = beta0.data

    W1 = torch.zeros(bsz, n_head, d_head, v_dim, device='cuda')
    print("Forwarding custom kernel...")
    out1 = fast_weight_delta(q1, k1, v1, beta1, W1)
    print("done.")

    # compute using torch
    q2 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    k2 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v2 = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    beta2 = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    q2.data = q0.data
    k2.data = k0.data
    v2.data = v0.data
    beta2.data = beta0.data

    # (len, B, H, dim)
    q_2 = q2.permute(2, 0, 1, 3)
    slen, bsz, n_head, d_head = q_2.shape

    q_2 = q_2.reshape(slen, bsz * n_head, d_head)

    k_2 = k2.permute(2, 0, 1, 3)
    k_2 = k_2.reshape(slen, bsz * n_head, d_head)

    v_2 = v2.permute(2, 0, 1, 3)
    v_2 = v_2.reshape(slen, bsz * n_head, v_dim)

    beta_2 = beta2.permute(2, 0, 1, 3)
    beta_2 = beta_2.reshape(slen, bsz * n_head, 1)

    W = torch.zeros(bsz * n_head, v_dim, d_head, device='cuda')

    out_list = []
    print("Forwarding PyTorch code...")
    for pos in range(slen):
        v_old = torch.bmm(W, k_2[pos].unsqueeze(2)).squeeze()
        v_insert = beta_2[pos] * (v_2[pos] - v_old)
        W = W + torch.bmm(v_insert.unsqueeze(2), k_2[pos].unsqueeze(1))
        out_t = torch.bmm(W, q_2[pos].unsqueeze(2)).squeeze()
        out_list.append(out_t.clone())
    print("done.")

    out2 = torch.stack(out_list)
    out2 = out2.view(slen, bsz, n_head, v_dim)
 
    out1 = out1.permute(2, 0, 1, 3)

    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"out1: {out1[s][b][h]}")
                print(f"out2: {out2[s][b][h]}")
                assert max_relative_error(
                    out1[s][b][h], out2[s][b][h]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Test Backward pass')
    print('##########################')

    # grad
    loss1 = out1.sum()
    q1.retain_grad()
    k1.retain_grad()
    v1.retain_grad()
    beta1.retain_grad()

    loss1.backward()

    loss2 = out2.sum()
    q2.retain_grad()
    k2.retain_grad()
    v2.retain_grad()
    beta2.retain_grad()

    loss2.backward()

    for s in range(slen):
        for b in reversed(range(bsz)):
            for h in range(n_head):
                print(f"s={s}, b={b}, h={h}")
                print(f"grad query1: {q1.grad[b][h][s]}")
                print(f"grad query2: {q2.grad[b][h][s]}")
                assert max_relative_error(
                    q1.grad[b][h][s], q2.grad[b][h][s]) < rel_threshold
                print("pass!")
   
                print(f"grad key1: {k1.grad[b][h][s]}")
                print(f"grad key2: {k2.grad[b][h][s]}")
                assert max_relative_error(
                    k1.grad[b][h][s], k2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad values1: {v1.grad[b][h][s]}")
                print(f"grad values2: {v2.grad[b][h][s]}")
                assert max_relative_error(
                    v1.grad[b][h][s], v2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad beta1: {beta1.grad[b][h][s]}")
                print(f"grad beta2: {beta2.grad[b][h][s]}")
                assert max_relative_error(
                    beta1.grad[b][h][s], beta2.grad[b][h][s]) < rel_threshold
                print("pass!")

    print("All tests pass.")
