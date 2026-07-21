#!/usr/bin/env python
"""M1/M2: chain speculative decoding micro-AL (study spec section 16).

M2 (--mode greedy): temperature-0 chain — draft proposes K greedy tokens,
the deployed target verifies sequentially by greedy agreement; tau =
accepted+1 with the target's token as the correction/bonus.

M1 (--mode t1): temperature-1 chain with CORRECT rejection sampling —
draft SAMPLES x_k ~ q_k; accept with prob min(1, p(x_k)/q(x_k)); on
rejection the correction token is drawn from normalize(max(p-q, 0)); on
full acceptance the bonus is drawn from the next p (standard speculative
sampling; Gate I toy tests in tests/test_rejection_sampling.py).

The draft is the RUNTIME adapter chain (same modules as the tree; draft
KV4 on appended K/V; re-prefilled per cycle on the visited prefix). The
target uses its persistent KVCache (target KV4 when t4kv4) with length-
pointer rollback — deployed cache semantics.

Records per cycle: tau, first-rejection depth, predicted alpha per depth
(M1), per-depth accept bools. Writes
shards/lkchain__<mode>__<tag>__<target>__<ds>.csv.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import torch.nn.functional as F
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.kv4_cache import install_kv4_on_past, fake_quant_kv
from eagle_spinquant.rejection_sampling import spec_step
from eagle_spinquant.eval_datasets import load_eval_prompts
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
TARGETS = {"t8": ("w8a8", 16), "t4": ("w4a4", 16), "t4kv4": ("w4a4", 4)}
D4P3 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True)


def hf_mask(T, Tc, device, dtype):
    m = torch.zeros(1, 1, T, Tc, device=device, dtype=dtype)
    if T > 1:
        m[0, 0] = torch.full((T, Tc), torch.finfo(dtype).min,
                             device=device, dtype=dtype).triu(Tc - T + 1)
    return m


class ChainDraft:
    """Runtime-adapter chain drafting: prefill on the visited prefix, then
    K recurrent one-token steps; returns tokens + full q logits."""

    def __init__(self, ea, adapter, dev, kv_bits=4):
        self.ea, self.ad, self.dev, self.kv = ea, adapter, dev, kv_bits

    @torch.no_grad()
    def propose(self, toks, a_vis, K, mode, gen):
        """EAGLE row alignment: N visited tokens -> N-1 prefill rows
        (e_{i+1}, a_i), i=0..N-2; the tail hidden a_{N-1} is unused."""
        ea, dev = self.ea, self.dev
        N = len(toks)
        assert a_vis.shape[0] == N
        T = N - 1
        E = ea.embed_tokens
        ids = torch.tensor(toks, device=dev)
        e_pref = E(ids[None, 1:]).half()               # (1, N-1, D)
        z = torch.cat([e_pref, a_vis[None, :T].half()], dim=-1)
        self.ad.split.select = "first"
        y = self.ad.split(z)
        pos = torch.arange(0, T, device=dev)[None]
        layer = ea.layers[0]
        lo = layer(y, attention_mask=hf_mask(T, T, dev, y.dtype),
                   position_ids=pos, past_key_value=None, use_cache=True)
        h_all, past = lo[0], lo[-1]
        if self.kv < 16:
            past = tuple(fake_quant_kv(p, bits=self.kv) for p in past)
        h_last = h_all[:, -1:]
        toks_out, qlogits = [], []
        for k in range(K):
            lg = self.ad.head(h_last.squeeze(1).half()).float()[0]
            if mode == "greedy":
                nxt = int(lg.argmax())
            else:
                nxt = int(torch.multinomial(F.softmax(lg, -1), 1,
                                            generator=gen))
            toks_out.append(nxt)
            qlogits.append(lg)
            if k == K - 1:
                break
            e_k = E(torch.tensor([[nxt]], device=dev)).half()
            z = torch.cat([e_k, h_last.half()], dim=-1)
            self.ad.split.select = "recurrent"
            y = self.ad.split(z)
            Tc = past[0].shape[2] + 1
            lo = layer(y, attention_mask=hf_mask(1, Tc, dev, y.dtype),
                       position_ids=torch.tensor([[T + k]], device=dev),
                       past_key_value=past, use_cache=True)
            h_all, past = lo[0], lo[-1]
            if self.kv < 16:
                nk = fake_quant_kv(past[0][:, :, -1:], bits=self.kv)
                nv = fake_quant_kv(past[1][:, :, -1:], bits=self.kv)
                past = (torch.cat([past[0][:, :, :-1], nk], 2),
                        torch.cat([past[1][:, :, :-1], nv], 2))
            h_last = h_all[:, -1:]
        return toks_out, qlogits


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lk-ckpt", default="shared")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="greedy", choices=["greedy", "t1"])
    ap.add_argument("--target", default="t4kv4", choices=list(TARGETS))
    ap.add_argument("--datasets", default="mtbench,c4,gsm8k")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    quant, t_kv = TARGETS[args.target]
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, quant, 0, device=dev, rotations_root=rr)
    bm = model.base_model
    bm.model.tree_mask = None
    past, _pd, cur_len = initialize_past_key_values(bm)
    if t_kv < 16:
        install_kv4_on_past(past, bits=t_kv)
    tok = eagle_bridge.get_tokenizer(model)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    R_T = stash["R1"].clone()
    sd0 = {k: v.detach().cpu().clone()
           for k, v in model.ea_layer.state_dict().items()}
    kw = dict(D4P3)
    ck = None
    if args.lk_ckpt == "shared":
        kw["embed_scale_alpha"] = 32.0
    else:
        ck = torch.load(args.lk_ckpt, map_location="cpu",
                        weights_only=False)
        stash = dict(stash)
        stash["R1"] = ck["R_D"].double()
        kw["first_fold_R"] = R_T
        kw["embed_scale_alpha"] = float(ck.get("alpha", 32.0))
    ad = ConcatSelectiveDraftAdapter(
        model, stash, dev, torch.float16, variant="folded",
        first_hidden_mode="gamma_R1", trace=False, **kw)
    ad.install()
    if ck is not None and "core_weights" in ck:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        from eval_eagle_lk_tree import inject_core
        inject_core(ad, model, ck, sd0, R_T, stash["gamma_f"],
                    stash["lm_head_weight"].float(), dev, 4)
    draft = ChainDraft(model.ea_layer, ad, dev, kv_bits=4)
    gen = torch.Generator(device=dev)
    eos = tok.eos_token_id

    for ds_name in args.datasets.split(","):
        out_csv = os.path.join(
            args.run_dir, "shards",
            f"lkchain__{args.mode}__{args.tag}__{args.target}__"
            f"{ds_name}.csv")
        if os.path.exists(out_csv):
            print(f"[lkchain] {args.tag}/{args.mode}/{ds_name}: exists, "
                  "skip", flush=True)
            continue
        prompts, _ = load_eval_prompts(ds_name, args.n_prompts, "eval")
        rows = []
        for p in prompts:
            gen.manual_seed(args.seed)
            ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            toks = ids[0].tolist()
            cur_len.zero_()
            h = bm.model(input_ids=ids, past_key_values=past,
                         use_cache=True)[0]
            a_vis = h[0]                       # (L, D) exposed basis
            p_next = bm.lm_head(h[:, -1:])[0, 0].float()
            n_new = 0
            taus, rejdep, alpha_pred = [], [], []
            while n_new < args.max_new_tokens:
                xk, qlg = draft.propose(toks, a_vis, args.K, args.mode,
                                        gen)
                base_len = len(toks)
                step = torch.tensor([xk], device=dev)
                h_k = bm.model(input_ids=step, past_key_values=past,
                               use_cache=True)[0]
                lg_k = bm.lm_head(h_k)[0].float()   # (K, V)
                p_list = [p_next] + [lg_k[i] for i in range(args.K - 1)]
                acc = 0
                corr = None
                al_cyc = []
                for k in range(args.K):
                    pk = F.softmax(p_list[k], -1)
                    qk = F.softmax(qlg[k], -1)
                    al_cyc.append(float(torch.minimum(pk, qk).sum()))
                    if args.mode == "greedy":
                        ok = int(pk.argmax()) == xk[k]
                        corr_tok = None if ok else int(pk.argmax())
                    else:
                        ok, corr_tok = spec_step(pk, qk, xk[k], gen=gen,
                                                 device=dev)
                    if ok:
                        acc += 1
                    else:
                        corr = corr_tok
                        break
                alpha_pred.append(al_cyc)
                if acc == args.K:
                    pK = F.softmax(lg_k[args.K - 1], -1)
                    corr = (int(pK.argmax()) if args.mode == "greedy"
                            else int(torch.multinomial(pK, 1,
                                                       generator=gen)))
                    rejdep.append(args.K + 1)
                else:
                    rejdep.append(acc + 1)
                # rollback target cache to visited + accepted, then feed
                # the correction/bonus token to restore the invariant
                keep = base_len + acc
                cur_len.fill_(keep)
                toks = toks + xk[:acc] + [corr]
                a_vis = torch.cat([a_vis, h_k[0, :acc]], dim=0)
                stepc = torch.tensor([[corr]], device=dev)
                h_c = bm.model(input_ids=stepc, past_key_values=past,
                               use_cache=True)[0]
                a_vis = torch.cat([a_vis, h_c[0]], dim=0)
                p_next = bm.lm_head(h_c)[0, 0].float()
                taus.append(acc + 1)
                n_new += acc + 1
                if eos in toks[-(acc + 1):]:
                    break
            rows.append(dict(
                tag=args.tag, mode=args.mode, target=args.target,
                dataset=ds_name, prompt_id=p["row_id"],
                acceptance_list=json.dumps(taus),
                first_rejection_depths=json.dumps(rejdep),
                alpha_pred=json.dumps(
                    [[round(a, 4) for a in c] for c in alpha_pred[:50]]),
                n_cycles=len(taus)))
        taus_all = [t for r in rows
                    for t in json.loads(r["acceptance_list"])]
        mal = sum(taus_all) / max(len(taus_all), 1)
        logging_utils.write_csv(out_csv, rows)
        print(f"[lkchain] {args.mode}/{args.tag}/{args.target}/{ds_name}: "
              f"micro-AL={mal:.4f} ({len(taus_all)} cycles)", flush=True)
    ad.uninstall()
    print(f"[lkchain] {args.tag} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
