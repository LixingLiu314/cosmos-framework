# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GR1 action-policy SFT with a Qwen3-VL-4B pretrained initialization.

This variant keeps the GR1 dataloader, optimizer allowlist, scheduler, and
policy-mode setup from ``gr1_robot_policy_merge``, but starts from Qwen3-VL-4B
HF weights instead of a Cosmos3-Nano DCP warm start:

* load the Qwen3-VL-4B reasoner/understanding pathway from
  ``QWEN_4B_MODEL_PATH``;
* copy the loaded reasoner weights into the generator pathway on fresh init;
* keep the reasoner frozen via the optimizer ``keys_to_select`` allowlist;
* leave VFM/action adapters randomly initialized; and
* train with action and vision loss weights both set to 1.0.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.defaults.vlm import create_qwen2_tokenizer_with_download, create_vlm_config
from cosmos_framework.configs.base.experiment.action.posttrain_config.gr1_robot_policy_merge import (
    gr1_robot_policy_merge,
)
from cosmos_framework.model.vfm.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


QWEN3VL_4B_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3VL_4B_MODEL_PATH = "${oc.env:QWEN_4B_MODEL_PATH,Qwen/Qwen3-VL-4B-Instruct}"
QWEN3VL_4B_CONFIG_JSON = "cosmos_framework/model/vfm/vlm/qwen3_vl/configs/Qwen3-VL-4B-Instruct.json"


gr1_robot_policy_merge_qwen3vl_4b = copy.deepcopy(gr1_robot_policy_merge)
gr1_robot_policy_merge_qwen3vl_4b["job"]["name"] = "gr1_robot_policy_merge_qwen3vl_4b"

_model_config = gr1_robot_policy_merge_qwen3vl_4b["model"]["config"]

# The reasoner is frozen by optimizer.keys_to_select in the parent recipe. Do
# not save it in train checkpoints; reload it from the Qwen 4B HF snapshot on
# resume so it remains the fixed pretrained reasoner.
_model_config["exclude_reasoner_weights_from_checkpoint"] = True

_model_config["vlm_config"] = dict(
    layer_module="Qwen2MoTDecoderLayer",
    model_name=QWEN3VL_4B_MODEL_NAME,
    tie_word_embeddings=True,
    use_system_prompt=False,
    pretrained_weights=dict(
        enabled=True,
        backbone_path=QWEN3VL_4B_MODEL_PATH,
        credentials_path="",
        enable_gcs_patch_in_boto3=False,
    ),
    model_instance=L(Qwen3VLTextForCausalLM)(
        config=L(create_vlm_config)(
            base_config=L(Qwen3VLMoTConfig.from_json_file)(
                json_file=QWEN3VL_4B_CONFIG_JSON,
            ),
            freeze_und=False,
            layer_module="MoTDecoderLayer",
            qk_norm_for_text=True,
            tie_word_embeddings=True,
        ),
    ),
    tokenizer=L(create_qwen2_tokenizer_with_download)(
        config_variant="hf",
        pretrained_model_name=QWEN3VL_4B_MODEL_PATH,
    ),
)

_rf_config = _model_config["rectified_flow_training_config"]
_rf_config["action_loss_weight"] = 1.0
_rf_config["image_loss_scale"] = 1.0
_rf_config["loss_scale"] = 1.0

_dataset_config = gr1_robot_policy_merge_qwen3vl_4b["dataloader_train"]["dataloader"]["datasets"]["gr1"]["dataset"]
_dataset_config["mode"] = "policy"

# Fresh Qwen-4B initialization: no 8B Cosmos3-Nano DCP warm start. With no
# load_path, OmniMoTModel.load_pretrained_model_if_needed loads the reasoner
# and then copies those weights into the generator pathway. VFM/action adapters
# keep their normal random initialization.
_checkpoint = gr1_robot_policy_merge_qwen3vl_4b["checkpoint"]
_checkpoint["keys_to_skip_loading"] = []
_checkpoint["load_path"] = ""
_checkpoint["load_training_state"] = False
_checkpoint["only_load_scheduler_state"] = False
_checkpoint["strict_resume"] = False


cs.store(
    group="experiment",
    package="_global_",
    name="gr1_robot_policy_merge_qwen3vl_4b",
    node=gr1_robot_policy_merge_qwen3vl_4b,
)
