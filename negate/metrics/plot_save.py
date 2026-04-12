# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Save and load plot data frame to JSON."""

import json

import pandas as pd
from pandas import Series
from pathlib import Path

from negate.io.spec import Spec, root_folder

plot_file = "plot_xp_data.json"


def save_frames(data_frame: pd.DataFrame, model_name: str) -> None:
    """Save dataframe to JSON with model name.

    :param data_frame: Input dataframe to serialize.
    :param model_name: Label for the model run.
    """

    data_frame["model_name"] = model_name
    frames = data_frame.to_dict(orient="records")
    data_log = str(root_folder / "results" / plot_file)
    Path(data_log).parent.mkdir(parents=True, exist_ok=True)
    with open(data_log, mode="tw+") as plot_data:
        json.dump(frames, plot_data, indent=4, ensure_ascii=False, sort_keys=False)


def load_frames(folder_path_name: str) -> tuple[pd.DataFrame, Series]:
    """Load dataframe and model name from JSON.

    :param folder_path_name: Subfolder under results containing plot data.
    :returns: Tuple of (dataframe, series of model names).
    """
    plot_path = root_folder / "results" / folder_path_name
    with open(str(plot_path / plot_file), "r") as plot_data:
        saved_frames = json.load(plot_data)
    xp_frames = pd.DataFrame.from_dict(json.loads(saved_frames))
    model_name = xp_frames.pop("model_name")
    return xp_frames, model_name
