"""
nnUNetTrainerMetaBottleneck: injects per-case metadata (days_post_stroke,
chronicity, site) at the ResidualEncoder bottleneck using MetaConditionedStage.

Metadata is read from imagesTr/*_0001.json sidecars at training time.
At inference time (nnUNetv2_predict without sidecars) MetaConditionedStage
falls back to pass-through — the network remains valid but unconditioned.

Dataset requirements:
  - dataset.json must declare 1 channel (T1w only).
  - imagesTr/ must contain *_0001.json sidecars produced by prepare_dataset.py.
  - metadata_stats.json must exist in the dataset root.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.meta_conditioning import MetaConditionedStage
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

log = logging.getLogger(__name__)


def _get_bottleneck_channels(network: torch.nn.Module) -> int:
    if hasattr(network, "encoder") and hasattr(network.encoder, "output_channels"):
        return network.encoder.output_channels[-1]
    stage = network.encoder.stages[-1]
    for m in reversed(list(stage.modules())):
        if isinstance(m, torch.nn.Conv3d):
            return m.out_channels
    raise RuntimeError("Cannot determine bottleneck channel count from network")


def _load_n_sites(dataset_name: str) -> int:
    stats_path = Path(nnUNet_preprocessed) / dataset_name / "metadata_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"metadata_stats.json required for nnUNetTrainerMetaBottleneck but not found at {stats_path}\n"
            f"Run prepare_dataset.py with the full dataset to generate it."
        )
    with stats_path.open() as f:
        n = len(json.load(f)["site"]["vocab"])
    log.info("Loaded n_sites=%d from %s", n, stats_path)
    return n


def _patch_bottleneck(network: torch.nn.Module, n_sites: int) -> MetaConditionedStage:
    """Replace encoder.stages[-1] with MetaConditionedStage; return the wrapper."""
    if not (hasattr(network, "encoder") and hasattr(network.encoder, "stages")):
        raise RuntimeError(
            "Network has no encoder.stages — MetaConditionedStage requires "
            "PlainConvUNet / ResidualEncoderUNet architecture."
        )
    channels = _get_bottleneck_channels(network)
    stage = network.encoder.stages[-1]
    conditioned = MetaConditionedStage(stage=stage, bottleneck_channels=channels, n_sites=n_sites)
    network.encoder.stages[-1] = conditioned
    log.info("Patched encoder.stages[-1]: %d channels, %d sites", channels, n_sites)
    return conditioned


class nnUNetTrainerMetaBottleneck(nnUNetTrainer):
    """
    Extends nnUNetTrainer with metadata conditioning at the encoder bottleneck.

    build_network_architecture (static) is also called by nnUNetPredictor at
    inference time, so the MetaConditionedStage is always present in the loaded
    weights regardless of whether metadata sidecars are available at that point.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._meta_lookup: dict[str, dict[str, Any]] = {}
        self._meta_stage: MetaConditionedStage | None = None
        self.num_epochs = 1000

    # ------------------------------------------------------------------
    # Network architecture
    # ------------------------------------------------------------------

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        if num_input_channels != 1:
            log.warning(
                "nnUNetTrainerMetaBottleneck expects 1 input channel (T1w); "
                "got %d — forcing 1.",
                num_input_channels,
            )
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            input_channels=1,
            output_channels=num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        n_sites = _load_n_sites(plans_manager.dataset_name)
        _patch_bottleneck(network, n_sites)
        return network

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        # Before calling super().initialize() (which loads checkpoints), validate
        # that the current metadata_stats.json site count matches what the network
        # was built for. This catches site vocab mismatches early instead of
        # letting them surface as cryptic RuntimeError during checkpoint load.
        # self._validate_metadata_vocab()
        super().initialize()
        self._meta_stage = self._find_meta_stage()
        self._load_case_metadata()

    def _validate_metadata_vocab(self) -> None:
        """Validate that network site_embed size matches current metadata_stats.json."""
        expected_n_sites = _load_n_sites(self.plans_manager.dataset_name)
        # Check encoder.stages[-1].meta_encoder.site_embed shape
        base = getattr(self.network, "_orig_mod", self.network)
        for module in base.modules():
            if isinstance(module, MetaConditionedStage):
                actual_n_sites = module.meta_encoder.site_embed.num_embeddings
                if actual_n_sites != expected_n_sites:
                    raise ValueError(
                        f"Site embedding size mismatch: network has {actual_n_sites} sites "
                        f"but metadata_stats.json declares {expected_n_sites}.\n"
                        f"This usually means a checkpoint was trained on a different system "
                        f"with fewer sites. Options:\n"
                        f"  1. Retrain folds 3-5 from scratch (recommended for consistency)\n"
                        f"  2. Copy metadata_stats.json from the other system if you have it\n"
                        f"  3. Use --c with non-strict checkpoint loading (loses site embeddings)"
                    )
                return

    def _find_meta_stage(self) -> MetaConditionedStage | None:
        """Locate MetaConditionedStage even if network has been torch.compiled."""
        base = getattr(self.network, "_orig_mod", self.network)
        for module in base.modules():
            if isinstance(module, MetaConditionedStage):
                return module
        log.warning("MetaConditionedStage not found in network — metadata disabled")
        return None

    def _load_case_metadata(self) -> None:
        preprocessed_dir = Path(nnUNet_preprocessed) / self.plans_manager.dataset_name
        images_dir = preprocessed_dir / "imagesTr"
        n = 0
        for json_path in sorted(images_dir.glob("*_0001.json")):
            case_id = json_path.name.replace("_0001.json", "")
            with json_path.open() as f:
                meta = json.load(f)
            self._meta_lookup[case_id] = {
                "days_norm": float(meta["days_norm"]),
                "chronicity": int(meta["chronicity"]),
                "site_idx": int(meta["site_idx"]),
            }
            n += 1
        log.info("Loaded metadata sidecars for %d cases from %s", n, images_dir)
        if n == 0:
            log.warning(
                "No *_0001.json found in %s — training without metadata conditioning",
                images_dir,
            )

    # ------------------------------------------------------------------
    # Metadata tensor assembly
    # ------------------------------------------------------------------

    def _meta_tensors(self, keys: list[str]) -> tuple[Tensor, Tensor, Tensor]:
        days_list, chron_list, site_list = [], [], []
        for key in keys:
            m = self._meta_lookup.get(key, {"days_norm": -1.0, "chronicity": 0, "site_idx": 0})
            if key not in self._meta_lookup:
                log.debug("No metadata for case '%s' — using sentinel values", key)
            days_list.append(m["days_norm"])
            chron_list.append(m["chronicity"])
            site_list.append(m["site_idx"])
        return (
            torch.tensor(days_list, dtype=torch.float32),
            torch.tensor(chron_list, dtype=torch.long),
            torch.tensor(site_list, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        keys = list(batch.get("keys", []))
        if self._meta_stage is not None and keys:
            days, chron, site_idx = self._meta_tensors(keys)
            # 5% site-unknown dropout: randomly mask site identity so the model
            # learns a useful representation for unseen sites at inference time.
            dropout_mask = torch.rand(site_idx.shape) < 0.05
            site_idx = site_idx.masked_fill(dropout_mask, 0)
            self._meta_stage.set_meta(days, chron, site_idx)
        try:
            return super().train_step(batch)
        finally:
            if self._meta_stage is not None:
                self._meta_stage.clear_meta()

    def validation_step(self, batch: dict) -> dict:
        keys = list(batch.get("keys", []))
        if self._meta_stage is not None and keys:
            self._meta_stage.set_meta(*self._meta_tensors(keys))
        try:
            return super().validation_step(batch)
        finally:
            if self._meta_stage is not None:
                self._meta_stage.clear_meta()

    # ------------------------------------------------------------------
    # Checkpointing — save MetaConditionedStage weights separately so
    # they can be loaded independently of the base network weights.
    # The companion file is <checkpoint_path>_meta.pth and contains only
    # the meta_encoder and proj keys (not the wrapped bottleneck stage).
    # ------------------------------------------------------------------

    def _meta_state_dict(self) -> dict:
        """Return state dict containing only MetaConditionedStage-specific weights."""
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        full = mod.state_dict()
        return {
            k: v for k, v in full.items()
            if "meta_encoder." in k or (
                # proj layer lives inside MetaConditionedStage — identified by
                # belonging to the same encoder stage as meta_encoder keys
                ".proj." in k and any(
                    k.rsplit(".proj.", 1)[0] == mk.rsplit(".meta_encoder.", 1)[0]
                    for mk in full if "meta_encoder." in mk
                )
            )
        }

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0 and not self.disable_checkpointing:
            meta_sd = self._meta_state_dict()
            if meta_sd:
                torch.save({"meta_weights": meta_sd}, filename + "_meta.pth")
                log.debug("Saved %d meta weight tensors to %s_meta.pth",
                          len(meta_sd), filename)


class nnUNetTrainerMetaBottleneck500epochs(nnUNetTrainerMetaBottleneck):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerMetaBottleneck1000epochs(nnUNetTrainerMetaBottleneck):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


class nnUNetTrainer500epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
