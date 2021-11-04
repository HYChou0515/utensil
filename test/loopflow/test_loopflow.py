import os
import sys
import unittest as ut
import warnings
from test.fixtures import FIXTURE_BASE

import pytest

from utensil.general.logger import get_logger

LOOPFLOW_INSTALLED = os.environ.get('LOOPFLOW_INSTALLED', '1') == '1'

logger = get_logger(__name__)


class TestSimpleFlow(ut.TestCase):

    @pytest.mark.timeout(10)
    @pytest.mark.xfail(
        condition=not LOOPFLOW_INSTALLED,
        reason="loopflow not installed",
        raises=ImportError,
    )
    @pytest.mark.skipif(
        condition=sys.platform == "darwin",
        reason="skipped for macos, may cause seg fault",
    )
    def test_end_to_end(self):
        from utensil.loopflow.functions import basic
        from utensil.loopflow.loopflow import (Flow,
                                               register_node_process_functions,
                                               reset_node_process_functions)

        from . import simple

        reset_node_process_functions()
        register_node_process_functions(proc_func_module=simple)
        register_node_process_functions(proc_funcs=[basic.GreaterThan])

        if os.path.isfile("simple.output"):
            warnings.warn("simple.output deleted")
            os.remove("simple.output")

        flow_path = os.path.join(FIXTURE_BASE, "simple.flow")
        flow = Flow.parse_yaml(flow_path)
        flow.start()

        self.assertTrue(os.path.isfile("simple.output"))
        import pickle
        with open("simple.output", "rb") as f:
            output = pickle.load(f)
        self.assertEqual(115, output)

        os.remove("simple.output")


class TestCovtypeFlow(ut.TestCase):

    @pytest.mark.timeout(60)
    @pytest.mark.xfail(
        condition=not LOOPFLOW_INSTALLED,
        reason="loopflow not installed",
        raises=ImportError,
    )
    @pytest.mark.skipif(
        condition=sys.platform == "darwin",
        reason="skipped for macos, may cause seg fault",
    )
    def test_end_to_end(self):
        from utensil.loopflow.functions import basic, dataflow
        from utensil.loopflow.loopflow import (Flow,
                                               register_node_process_functions,
                                               reset_node_process_functions)
        reset_node_process_functions()
        register_node_process_functions(proc_func_module=basic)
        register_node_process_functions(proc_func_module=dataflow)

        flow_path = os.path.join(FIXTURE_BASE, "covtype.flow")
        flow = Flow.parse_yaml(flow_path)
        results = flow.start()
        logger.debug(results)
        self.assertEqual(2, len(results))
        for result in results:
            self.assertEqual("TEST_SCORE", result[0])
            self.assertEqual(1, len(result[1]))
            self.assertEqual(3, len(result[1][0]))
            self.assertEqual("ACCURACY", result[1][0][0])
            self.assertEqual("TEST_DATA", result[1][0][1])
            self.assertGreaterEqual(1, result[1][0][2])
            self.assertLessEqual(0, result[1][0][2])


if __name__ == "__main__":
    ut.main()
