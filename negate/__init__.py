# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# ruff: noqa

from datasets import logging as ds_logging, disable_progress_bars as ds_disable_progress_bars
from huggingface_hub import logging as hf_logging
from huggingface_hub.utils.tqdm import disable_progress_bars as hf_disable_progress_bars
from transformers import logging as tf_logging
from diffusers.utils import logging as df_logging
from timm.utils.log import setup_default_logging
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
setup_default_logging(logging.ERROR)
for logger in [df_logging, ds_logging, hf_logging, tf_logging]:
    logger.set_verbosity_error()
    ds_disable_progress_bars()
    hf_disable_progress_bars()

from negate.io.blurb import Blurb
from negate.io.config import root_folder
from negate.io.spec import Spec, load_spec, load_metadata
from negate.metrics.track import chart_decompositions, run_training_statistics
from negate.train import build_train_call, pretrain, train_model, training_loop
from negate.io.save import save_train_result, end_processing, save_features
from negate.metrics.heuristics import compute_weighted_certainty
from negate.inference import infer_origin, InferContext
