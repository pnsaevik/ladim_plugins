
def test_snapshot():
    import ladim_plugins.tests.test_examples
    import os
    os.chdir(os.path.dirname(ladim_plugins.tests.test_examples.__file__))
    ladim_plugins.tests.test_examples.test_output_matches_snapshot('shrimp')
