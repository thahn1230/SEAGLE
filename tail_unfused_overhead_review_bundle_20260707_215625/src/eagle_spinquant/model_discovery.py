"""Resolve model candidates and verify the EAGLE-1 x SpinQuant intersection.

Static checks (architecture rotatable by SpinQuant, hidden/vocab agreement between
target and EAGLE draft) plus optional HF config fetches. Never downloads weights;
uses the local HF cache when present. Used by scripts/01.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDIDATES_YAML = os.path.join(PROJECT_ROOT, "configs", "model_candidates.yaml")

# SpinQuant's rotation/PTQ path is written against LLaMA-class decoder-only models.
SPINQUANT_ROTATABLE_ARCHS = {"LlamaForCausalLM"}


@dataclass
class Candidate:
    candidate_name: str
    hf_target_model: str
    hf_eagle_draft_or_training_source: str
    architecture: str
    eagle_support_status: str
    spinquant_support_status: str
    recommended_initial_candidate: bool = False
    reason: str = ""
    checks: dict = field(default_factory=dict)


def load_candidates(path: str = CANDIDATES_YAML) -> list[Candidate]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Candidate(**{k: v for k, v in c.items() if k in Candidate.__dataclass_fields__})
            for c in data["candidates"]]


def _local_snapshot_dir(repo_id: str) -> str | None:
    """Return a local HF cache snapshot dir for repo_id if fully present."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    folder = "models--" + repo_id.replace("/", "--")
    snaps = os.path.join(cache, folder, "snapshots")
    if not os.path.isdir(snaps):
        return None
    for name in sorted(os.listdir(snaps)):
        d = os.path.join(snaps, name)
        if os.path.isfile(os.path.join(d, "config.json")):
            return d
    return None


def fetch_config(repo_id: str, allow_network: bool = True) -> dict | str:
    """Return config.json as dict, or an error tag string."""
    local = _local_snapshot_dir(repo_id)
    if local:
        with open(os.path.join(local, "config.json")) as f:
            return json.load(f)
    if not allow_network:
        return "not-cached-offline"
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return "no-huggingface-hub"
    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except GatedRepoError:
        return "gated-needs-auth"
    except RepositoryNotFoundError:
        return "repo-not-found"
    except Exception as e:
        return f"fetch-error:{type(e).__name__}"
    with open(path) as f:
        return json.load(f)


def weights_present(repo_id: str) -> bool:
    """True if the local cache holds actual weight shards (not just config)."""
    d = _local_snapshot_dir(repo_id)
    if not d:
        return False
    for name in os.listdir(d):
        if name.endswith(".safetensors") or name.endswith(".bin"):
            # resolve symlink size
            p = os.path.realpath(os.path.join(d, name))
            if os.path.isfile(p) and os.path.getsize(p) > 10 * 2**20:  # >10MB = real shard
                return True
    return False


def check_candidate(cand: Candidate, allow_network: bool = True) -> Candidate:
    checks: dict[str, Any] = {}
    checks["architecture_rotatable"] = cand.architecture in SPINQUANT_ROTATABLE_ARCHS
    checks["target_weights_cached"] = weights_present(cand.hf_target_model)
    draft_id = cand.hf_eagle_draft_or_training_source
    is_repo = draft_id.count("/") == 1 and " " not in draft_id
    checks["draft_is_repo"] = is_repo
    checks["draft_weights_cached"] = weights_present(draft_id) if is_repo else False

    tgt = fetch_config(cand.hf_target_model, allow_network)
    checks["target_config"] = tgt if isinstance(tgt, str) else "ok"
    drf = fetch_config(draft_id, allow_network) if is_repo else "training-path-only"
    checks["draft_config"] = drf if isinstance(drf, str) else "ok"

    if isinstance(tgt, dict):
        checks["target_hidden_size"] = tgt.get("hidden_size")
        checks["target_vocab_size"] = tgt.get("vocab_size")
        checks["target_num_heads"] = tgt.get("num_attention_heads")
        checks["target_rope_theta"] = tgt.get("rope_theta", 10000.0)
        checks["target_tie_word_embeddings"] = tgt.get("tie_word_embeddings", False)
    if isinstance(tgt, dict) and isinstance(drf, dict):
        checks["hidden_size_match"] = tgt.get("hidden_size") == drf.get("hidden_size")
        checks["vocab_size_match"] = tgt.get("vocab_size") == drf.get("vocab_size")
    cand.checks = checks
    return cand


def resolve_target_and_draft(cfg: dict) -> dict:
    """Given default_experiment.yaml's `model` + `paths`, return resolved local
    paths (or repo ids) for target and draft, preferring explicit local paths,
    then the HF cache, then the bare repo id (HF will resolve at load time)."""
    model = cfg.get("model", {})
    paths = cfg.get("paths", {})
    target_id = model.get("target")
    draft_id = model.get("eagle_draft")
    target = paths.get("target_weights_local") or _local_snapshot_dir(target_id) or target_id
    draft = paths.get("draft_weights_local") or _local_snapshot_dir(draft_id) or draft_id
    return {
        "target_id": target_id, "draft_id": draft_id,
        "target_path": target, "draft_path": draft,
        "target_weights_cached": weights_present(target_id),
        "draft_weights_cached": weights_present(draft_id),
    }
