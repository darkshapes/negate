# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from datasets import Dataset
from negate import features, build_datasets, grade, TrainResult, in_console


def main():
    dataset: Dataset = build_datasets()

    features_dataset: Dataset = features(dataset)

    train_result: TrainResult = grade(features_dataset)

    in_console(train_result)


if __name__ == "__main__":
    main()
