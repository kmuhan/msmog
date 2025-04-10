import os, sys
import argparse
import math, random
import torch
import tqdm
import time

from custom_gates import *


def _train_step(model, load_balance, X, Y, h_cache, eval_only, loss_div=1):
    """Single training step."""

    out, h_cache = model(X, h_cache)
    out = out.view(-1, out.size(-1))
    loss = torch.nn.functional.nll_loss(out, Y.view(-1))
    loss_value = loss.item() / loss_div

    if not eval_only:
        # loss term from adaptive-span
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            loss += sum(
                model.module.layers[layer_i].attn.attn.adaptive_span.get_loss()
                for layer_i in range(model.module.attn_layer_count)
            )

        if load_balance > 0:
            balance_loss = 0
            for name, m in model.named_modules():
                if isinstance(m, CustomNaiveGate_Balance_SMoE) or isinstance(
                    m, CustomNaiveGate_Balance_XMoE
                ):
                    if m.loss is not None:
                        balance_loss += m.loss
            loss += load_balance * balance_loss
        (loss / loss_div).backward(retain_graph=True)
    return loss_value, h_cache


def _train_batch(
    model, load_balance, optimizer, scheduler, X, Y, h_cache, eval_only, batch_split
):
    """Train on a batch."""

    optimizer.zero_grad()

    if batch_split == 1:
        # process a batch in a single step (default behaviour)
        loss_value, h_cache = _train_step(model, load_balance, X, Y, h_cache, eval_only)
    else:
        # split a batch into multiple pieces that each can fit in memory
        assert X.size(0) % batch_split == 0
        split_size = X.size(0) // batch_split
        loss_value = 0
        h_cache_list = []
        for split_ind in range(batch_split):
            split_slice = slice(split_ind * split_size, (split_ind + 1) * split_size)
            split_h_cache = [h[split_slice, :, :] for h in h_cache]
            split_loss_value, split_h_cache = _train_step(
                model,
                load_balance,
                X[split_slice, :],
                Y[split_slice],
                split_h_cache,
                eval_only,
                batch_split,
            )
            loss_value += split_loss_value
            h_cache_list.append(split_h_cache)
        h_cache = [
            torch.cat([h_cache_list[i][l] for i in range(batch_split)], dim=0)
            for l in range(len(h_cache))
        ]
    if not eval_only:
        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # make sure span parameters are in a correct range
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            for layer in model.module.layers:
                if layer.use_attn:
                    layer.attn.attn.adaptive_span.clamp_param()
    return loss_value, h_cache


def train_iteration(
    model,
    load_balance,
    optimizer,
    scheduler,
    data,
    nb_batches_per_iter,
    block_size,
    eval_only,
    train_pos,
    h_cache,
    batch_split,
    checkpoint_path,
):
    """Single training iteration."""
    if eval_only:
        model.eval()
    else:
        model.train()

    nb_batches_per_iter_max = nb_batches_per_iter
    if eval_only:
        # eval on fewer batches during training for speed-up
        nb_batches_per_iter_max = max(1, nb_batches_per_iter // 10)
        nb_batches_per_iter_max = min(
            nb_batches_per_iter_max, math.ceil(data.size(1) / block_size)
        )

    loss_all = 0
    actual_nb_batches_per_iter = 0
    for _ in tqdm.tqdm(range(nb_batches_per_iter_max)):
        actual_nb_batches_per_iter += 1
        X = data[:, train_pos : train_pos + block_size].contiguous()
        Y = data[:, train_pos + 1 : train_pos + block_size + 1].contiguous()

        loss, h_cache = _train_batch(
            model=model,
            load_balance=load_balance,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X,
            Y=Y,
            h_cache=h_cache,
            eval_only=eval_only,
            batch_split=batch_split,
        )
        loss_all += loss
        train_pos += block_size
        if train_pos >= data.size(1) - block_size:
            # reached the end. randomize the offset to reduce overfitting
            train_pos = random.randrange(block_size)
            # reset the cache
            for h in h_cache:
                h.fill_(0)

    loss_all = loss_all / actual_nb_batches_per_iter
    return loss_all, train_pos, h_cache

from ptflops import get_model_complexity_info

def get_flops_and_params(model, batch_size, block_size, hidden_size, device="cuda"):
    """
    model: nn.Module (wrapped with DDP or not)
    batch_size: int
    block_size: int (sequence length)
    hidden_size: int
    device: str ("cuda" or "cpu")
    """
    model.eval()
    model = model.module if hasattr(model, "module") else model

    # Dummy inputs (X, h_cache)
    X = torch.zeros(batch_size, block_size).long().to(device)

    h_cache = [
        torch.zeros(
            batch_size,
            model.layers[layer_i].attn.attn.get_cache_size(),
            hidden_size
        ).to(device)
        for layer_i in range(model.attn_layer_count)
    ]

    def input_constructor(input_res):
        return (X, h_cache)

    input_shape = (block_size,)  # sequence length only

    with torch.cuda.device(0 if device == "cuda" else -1):
        macs, params = get_model_complexity_info(
            model,
            input_res=input_shape,
            input_constructor=input_constructor,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )

    print(f"##### FLOPs: {macs}")
    print(f"##### Params: {params}")
    return macs, params


# do full evaluation
def full_eval(model, optimizer, scheduler, data, block_size, hidden_size):
    model.eval()
    train_pos = 0
    nb_batches_per_iter_max = math.ceil(data.size(1) / block_size)
    h_cache = [
        torch.zeros(
            data.size(0),
            model.module.layers[layer_i].attn.attn.get_cache_size(),
            hidden_size,
        ).to(data.device)
        for layer_i in range(model.module.attn_layer_count)
    ]

    loss_all = 0
    actual_nb_batches_per_iter = 0
    total_time = 0
    flops_printed = False
    
    for _ in tqdm.tqdm(range(nb_batches_per_iter_max)):
        actual_nb_batches_per_iter += 1
        X = data[:, train_pos : train_pos + block_size].contiguous()
        Y = data[:, train_pos + 1 : train_pos + block_size + 1].contiguous()

        # calculate time
        start_time = time.time()

        if not flops_printed:
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    with_flops=True,
                    profile_memory=True,
                    record_shapes=True
                ) as prof:
                    with torch.no_grad():
                        _ = model.module(X, h_cache)

                # 전체 FLOPs
                total_flops = sum([evt.flops for evt in prof.key_averages() if evt.flops is not None])
                print(f"\n##### Total FLOPs (per forward pass): {total_flops / 1e9:.3f} GFLOPs")

                # top 연산 기준 출력
                print("\n##### FLOPs and time (Top 10 ops by FLOPs):")
                print(prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=10
                ))

                # layer별 FLOPs 정리
                from collections import defaultdict
                flops_per_op = defaultdict(float)
                for evt in prof.key_averages():
                    if evt.flops is not None:
                        flops_per_op[evt.key] += evt.cuda_time_total

                print("\n##### FLOPs per operator (in GFLOPs):")
                for key, flops in sorted(flops_per_op.items(), key=lambda x: -x[1]):
                    print(f"{key:40s}: {flops / 1e9:.3f} GFLOPs")

                flops_printed = True

            except Exception as e:
                print("##### FLOPs estimation failed due to:")
                print(f"{type(e)} : {e}")
                flops_printed = True

        loss, h_cache = _train_batch(
            model=model,
            load_balance=0,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X,
            Y=Y,
            h_cache=h_cache,
            eval_only=True,
            batch_split=1,
        )

        end_time = time.time()

        total_time += end_time - start_time

        loss_all += loss
        train_pos += block_size
        if train_pos >= data.size(1) - block_size:
            # Skip the remaining tokens as it can't make a whole block.
            # An effect on performance should be negligable for a large data.
            break

    print("#####inference time: ", total_time)

    loss_all = loss_all / actual_nb_batches_per_iter
    return loss_all
