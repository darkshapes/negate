# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from datasets import Dataset
from negate import features, build_datasets, grade, TrainResult, in_console, save_model
from sys import argv


def main():
    match argv[-1]:
        case "train":
            dataset: Dataset = build_datasets()

            features_dataset: Dataset = features(dataset)

            train_result: TrainResult = grade(features_dataset)
            save_model(train_result)

            in_console(train_result)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
