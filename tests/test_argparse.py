# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import PosixPath
from unittest.mock import patch
from argparse import Namespace


@patch("negate.extract.ResidualExtractor")
def test_main_correct_args(mock_run):
    """Test if main() correctly parses arguments and calls the extraction process."""
    from negate.extract import main

    # Mocked arguments
    mock_args = Namespace(input="/path/to/input", output="/path/to/output", verbose=True, graph=True)
    mock_args_residual = PosixPath("/path/to/input"), PosixPath("/path/to/output"), True

    # Call main with mocked arguments
    with patch("negate.extract.argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = mock_args
        try:
            main()
        except (AttributeError, TypeError) as _:
            pass

    # residual_extractor = ResidualExtractor(input_folder, output_folder, verbose)
    # Verify the arguments were parsed correctly
    mock_parse.assert_called_once()

    # Verify the correct initialization
    assert mock_args.input == "/path/to/input"
    assert mock_args.output == "/path/to/output"
    assert mock_args.verbose is True
    assert mock_args.graph is True

    # Verify asyncio.run was called with the correct async function
    mock_run.assert_called_once_with(*mock_args_residual)
