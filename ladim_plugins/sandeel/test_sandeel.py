from ..tests import test_examples


def test_snapshot():
    test_examples.test_output_matches_snapshot('sandeel')
