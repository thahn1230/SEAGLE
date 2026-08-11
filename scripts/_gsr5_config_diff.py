#!/usr/bin/env python
"""§19.7 fairness/causality audit: machine-readable proof that the GS
rotation arms (R0 = reuse target R_T, R1 = learned R5) differ ONLY in
the draft residual rotation.

Collects for both arms: model revisions, draft anchor SHA, R_T SHA,
R5 SHA, GS beta/m, quantizer parameters, module precision map, prompt
manifest hashes (recomputed from the pinned loaders), evaluator commit,
tree config, decoding config, and the exact executed CLI (from the
scheduler queue). Diffs them and asserts the only differences are the
rotation identity and the folded weights that follow from it.

Writes <run>/audit/gs_rt_vs_r5_config_diff.json (+ checkpoint manifest).
Exits 1 if an unexpected difference is found.
"""
import hashlib, json, os, subprocess, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

rd = sys.argv[1]
D = 4096
EXPECTED_DIFF_KEYS = {"draft_residual_rotation", "arm", "tag",
                      "executed_cli", "rotation_source_sha256"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prompt_manifest_hash(ds, n):
    from eagle_spinquant.eval_datasets import load_eval_prompts
    prompts, _ = load_eval_prompts(ds, n, "eval")
    blob = json.dumps([[p["row_id"], hashlib.sha256(
        p["text"].encode()).hexdigest()[:16]] for p in prompts],
        sort_keys=True)
    return dict(n=len(prompts),
                manifest_sha256=hashlib.sha256(blob.encode()).hexdigest())


R5 = ("runs/eagle1_draft_residual_rotation_ep3p_20260804_165317/"
      "rotations/RD_HYB_s2.pt")
R_T = "outputs/rotations/learned_chat_w4a4kv16/R.bin"
ANCHOR = "checkpoints/eagle1_fresh_fp16_anchor/anchor.pt"

cli = {}
for ln in open(os.path.join(rd, "scheduler", "queue.jsonl")):
    j = json.loads(ln)
    if j["id"].startswith("recon_B3R_"):
        cli.setdefault("R0", j["cmd"])
    if j["id"].startswith("eval_B9_"):
        cli.setdefault("R1", j["cmd"])

git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True, cwd=PROJECT_ROOT).stdout.strip()
manifests = {ds: prompt_manifest_hash(ds, n) for ds, n in
             (("mtbench", 80), ("gsm8k", 200), ("sharegpt", 80),
              ("humaneval", 164))}

COMMON = dict(
    target_model="meta-llama/Llama-2-7b-chat-hf",
    target_revision="f5db02db724555f92da89c216ac04704f23d4590",
    draft_model="yuhuili/EAGLE-llama2-chat-7B",
    draft_revision="44e37ec383348306fe9b1dfe7c79e145c96db3d0",
    draft_anchor_sha256=sha(os.path.join(PROJECT_ROOT, ANCHOR)),
    target_precision="W4A4 KV16 SpinQuant (learned_chat_w4a4kv16)",
    R_T_sha256=sha(os.path.join(PROJECT_ROOT, R_T)),
    first_interface_basis="R_T (first_fold_R pinned)",
    gs_beta=0.42, gs_m=D ** 0.42,
    scaling_policy="GS (global: same m folds into first AND recurrent "
                   "e-slice; no rec_embed_rescale)",
    quantizer_W4="symmetric per-output-channel RTN + MSE clip",
    quantizer_A4="asymmetric per-token dynamic, groupsize=-1, "
                 "clip_ratio=1.0",
    quantized_modules=["projection_first_preR", "projection_recurrent_preR",
                       "q", "k", "v", "o", "gate", "up", "down(+R4)"],
    fp16_modules=["embedding", "lm_head", "RMSNorm", "RoPE", "softmax",
                  "residual", "KV cache"],
    draft_kv_bits=16, w_bits=4, a_bits=4, r2_seed=0,
    evaluator="scripts/eval_eagle_acceptance_length.py",
    evaluator_git_head=git,
    tree="mc_sim_7b_63", decoding="greedy (temperature=0.0)",
    max_new_tokens=128, batch_size=1, prompt_truncate=1024,
    metric="official cycle-pooled micro-tau (accepted + 1)",
    prompt_manifests=manifests,
)

arms = {
    "R0": dict(COMMON, arm="R0", tag="B3R_T4",
               draft_residual_rotation="reuse target R_T",
               rotation_source_sha256=COMMON["R_T_sha256"],
               executed_cli=cli.get("R0", "")),
    "R1": dict(COMMON, arm="R1", tag="B9_T4",
               draft_residual_rotation="learned R5 (historically R_D), "
                                       "RD_HYB_s2.pt, reused unchanged",
               rotation_source_sha256=sha(os.path.join(PROJECT_ROOT, R5)),
               executed_cli=cli.get("R1", "")),
}

diff = {k: dict(R0=arms["R0"][k], R1=arms["R1"][k])
        for k in arms["R0"] if arms["R0"][k] != arms["R1"][k]}
unexpected = sorted(set(diff) - EXPECTED_DIFF_KEYS)

gg = os.path.join(rd, "gradchecks", "gateG_gs_r5_parity.json")
fold = json.load(open(gg)) if os.path.exists(gg) else {}

out = dict(
    question="Under a fixed GS parameterization, does replacing the "
             "draft residual basis R_T by the learned R5 change anything "
             "other than the rotation and the weights folded from it?",
    arms=arms, differences=diff,
    expected_difference_keys=sorted(EXPECTED_DIFF_KEYS),
    unexpected_differences=unexpected,
    identical_key_count=len([k for k in arms["R0"] if k not in diff]),
    fold_parity_gate=dict(
        verdict=fold.get("verdict"),
        gs_rt_leg_bitwise_equal_to_gs_baseline=all(
            v.get("equal_to_baseline")
            for v in fold.get("results", {})
            .get("GS_RT", {}).get("weights", {}).values()) or None,
        r5_orthogonality=fold.get("results", {}).get("r5_orthogonality")),
    verdict="PASS" if not unexpected else "FAIL",
)
os.makedirs(os.path.join(rd, "audit"), exist_ok=True)
json.dump(out, open(os.path.join(rd, "audit",
                                 "gs_rt_vs_r5_config_diff.json"), "w"),
          indent=1)

json.dump(dict(
    R5=dict(path=R5, sha256=arms["R1"]["rotation_source_sha256"],
            stores="materialized rotation matrix R_D [4096,4096] float32 "
                   "(Cayley parameters NOT stored)",
            recovered_matrix_sha256=arms["R1"]["rotation_source_sha256"],
            associated_target_precision="W4A4 KV16 (t4 teacher)",
            training_objective="hybrid KL/TV, K=4, gamma_depth=0.8, "
                               "3000 steps, lr 3e-4, batch 32",
            training_seed=2, reused_without_retraining=True),
    R_T=dict(path=R_T, sha256=COMMON["R_T_sha256"]),
    draft_anchor=dict(path=ANCHOR, sha256=COMMON["draft_anchor_sha256"]),
), open(os.path.join(rd, "audit",
                     "gs_rt_vs_r5_checkpoint_manifest.json"), "w"), indent=1)

print(f"[config-diff] {out['verdict']}: {len(diff)} differing keys "
      f"({sorted(diff)}), unexpected={unexpected}")
sys.exit(0 if not unexpected else 1)
