# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for CLI help text."""

import argparse
import sys
from io import StringIO
from negate.extract.unified_core import ExtractionModule


def test_process_help_includes_all_modules():
    """Verify process command help lists all ExtractionModule options."""
    from negate.__main__ import _load_blurb_text, _load_model_choices, _build_parser

    blurb = _load_blurb_text()
    choices = _load_model_choices()

    parser = _build_parser(
        blurb=blurb,
        choices=choices,
        list_results=[],
        list_model=[],
        inference_pair=[],
    )

    help_output = StringIO()
    try:
        parser.parse_args(["process", "-h"])
    except SystemExit:
        pass

    help_text = parser.format_help()
    process_subparser = None

    for action in parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            if "process" in action.choices:
                process_subparser = action.choices["process"]
                break

    assert process_subparser is not None, "process subparser not found"

    process_help = process_subparser.format_help()

    for module in ExtractionModule:
        assert module.name in process_help, f"Module {module.name} not found in process help text"


def test_process_help_module_list_matches_enum():
    """Verify the module list in help matches ExtractionModule enum."""
    from negate.__main__ import _load_blurb_text, _load_model_choices, _build_parser

    blurb = _load_blurb_text()
    choices = _load_model_choices()

    parser = _build_parser(
        blurb=blurb,
        choices=choices,
        list_results=[],
        list_model=[],
        inference_pair=[],
    )

    for action in parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            if "process" in action.choices:
                process_subparser = action.choices["process"]
                break

    process_help = process_subparser.format_help()

    module_names = [mod.name for mod in ExtractionModule]
    for module_name in module_names:
        assert module_name in process_help, f"Module {module_name} missing from help"

    assert len(module_names) == len(ExtractionModule), "Module list count mismatch"
