import multiprocessing
import os
import sys
import unittest as ut
import warnings
from test.fixtures import FIXTURE_BASE
from unittest.mock import patch

import pytest

from utensil.general.logger import get_logger
from utensil.loopflow import loopflow

LOOPFLOW_INSTALLED = os.environ.get("LOOPFLOW_INSTALLED", "1") == "1"

logger = get_logger(__name__)


class TestParseNodeTask(ut.TestCase):

    def test_should_fail_parse_dict_with_more_than_1_items(self):
        with pytest.raises(
                SyntaxError,
                match=
            ("Task specified by a dict should have exactly a pair of key and"
             " value, got .*"),
        ):
            loopflow.NodeTask.parse({"A": 1, "B": 2})

    def test_should_fail_parse_types_other_than_dist_and_str(self):
        with pytest.raises(
                SyntaxError,
                match="Task item support parsed by only str and dict, got",
        ):
            loopflow.NodeTask.parse(3)

    # pylint: disable=abstract-class-instantiated
    @patch.object(loopflow.NodeTask, "__abstractmethods__", set())
    def test_cannot_be_called_directly(self):
        task = loopflow.NodeTask()
        with pytest.raises(NotImplementedError):
            task()

    def test_str(self):

        class MyTask(loopflow.NodeTask):

            def main(self):
                return 1

        loopflow.reset_node_tasks()
        loopflow.register_node_tasks(task_map={"MY_TASK": MyTask})
        task = loopflow.NodeTask.parse("MY_TASK")
        self.assertEqual(1, task[0]())

    def test_dict_in_dict(self):

        class MyTask(loopflow.NodeTask):

            def __init__(self, *args, a=1, b=2, c=3, d=4, **kwargs):
                super().__init__(*args, **kwargs)
                self.a = a
                self.b = b
                self.c = c
                self.d = d

            def main(self):
                return self.a, self.b, self.c, self.d

        loopflow.reset_node_tasks()
        loopflow.register_node_tasks(task_map={"MY_TASK": MyTask})
        task = loopflow.NodeTask.parse({"MY_TASK": {"a": 10, "d": 30, "c": 12}})
        self.assertEqual((10, 2, 12, 30), task[0]())

    def test_str_in_dict(self):

        class MyTask(loopflow.NodeTask):

            def __init__(self, x, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.x = x

            def main(self):
                return self.x + 10

        loopflow.reset_node_tasks()
        loopflow.register_node_tasks(task_map={"MY_TASK": MyTask})
        task = loopflow.NodeTask.parse({"MY_TASK": 5})
        self.assertEqual(15, task[0]())

    def test_list_in_dict(self):

        class MyTask(loopflow.NodeTask):

            def __init__(self, *summed, **kwargs):
                super().__init__(*summed, **kwargs)
                self.summed = summed

            def main(self):
                return sum(self.summed)

        loopflow.reset_node_tasks()
        loopflow.register_node_tasks(task_map={"MY_TASK": MyTask})
        task = loopflow.NodeTask.parse({"MY_TASK": [1, 2, 3]})
        self.assertEqual(6, task[0]())


class TestNodeWorkerBuilder(ut.TestCase):

    def test_should_fail_when_meta_is_None(self):
        with pytest.raises(
                RuntimeError,
                match="meta should be defined before build a node worker",
        ):
            loopflow.NodeWorkerBuilder().build()


class TestNodeWorker(ut.TestCase):

    def test_exception_block(self):

        class MyTask(loopflow.NodeTask):

            def main(self):
                raise Exception("dummy exception")

        end_q = multiprocessing.SimpleQueue()
        result_q = multiprocessing.SimpleQueue()
        meta = loopflow.NodeMeta("", [], [], [MyTask()], tuple(), result_q,
                                 end_q)
        worker = loopflow.NodeWorker(meta)
        with patch("utensil.loopflow.loopflow.logger.exception") as mock:
            with pytest.raises(Exception, match="dummy exception") as e:
                worker.run()
                assert not end_q.empty()
                mock.assert_called_once_with(e)


class TestParentSpecifierToken(ut.TestCase):

    def test_bad_string_format(self):
        s = "A.B/C=D/E=F,G"
        with pytest.raises(
                SyntaxError,
                match=
                f"Each ParentSpecifierToken should match regex=.*, got {s}",
        ):
            loopflow.ParentSpecifierToken.parse_one(s)


class TestParents(ut.TestCase):

    def test_fail_on_duplicated_parent_keys(self):
        kwargs = {0: ["A|B", "C"]}
        args = [["D|E", "F"]]
        with pytest.raises(
                SyntaxError,
                match=(
                    "Parent key represents where the param passed from a parent"
                    f" goes to, so it must be unique. Got multiple '{0}'"),
        ):
            loopflow.Parents(args, kwargs)


class TestSenders(ut.TestCase):

    def test_fail_given_list_of_list(self):
        with pytest.raises(
                SyntaxError,
                match=
            (r"Senders do not support parsed by list of list, got '\['A'\]'"
             r" in a list"),
        ):
            loopflow.Senders.parse([["A"], ["B"]])

    def test_fail_given_except_str_dict_and_list(self):
        with pytest.raises(
                SyntaxError,
                match=
                "Senders support parsed by only dict, list and str, got '3'",
        ):
            loopflow.Senders.parse(3)


class TestCallers(ut.TestCase):

    def test_parsed_by_list(self):
        c = loopflow.Callers(["FOO", "BAR", "BAZ"])
        self.assertEqual(c.parent_keys, {0, 1, 2})
        self.assertEqual(c.node_map["FOO"][0][0], 0)
        self.assertEqual(c.node_map["BAR"][0][0], 1)
        self.assertEqual(c.node_map["BAZ"][0][0], 2)

    def test_fail_given_dict(self):
        with pytest.raises(
                SyntaxError,
                match="Callers support parsed by only str and list, got '.*'",
        ):
            loopflow.Callers.parse({"A": 1})


class TestNode(ut.TestCase):

    def test_exception_block(self):

        def main(self):
            raise Exception("dummy exception")

        end_q = multiprocessing.SimpleQueue()
        node = loopflow.Node("n", False, [], end_q,
                             multiprocessing.SimpleQueue())
        with patch("utensil.loopflow.loopflow.Node.main",
                   main), patch("utensil.loopflow.loopflow.logger.exception"
                               ) as logger_exception_mock, pytest.raises(
                                   Exception, match="dummy exception") as err:
            node.run()
            logger_exception_mock.assert_called_once_with(err)
            self.assertTrue(not end_q.empty())

    def test_fail_when_given_name_as_SWITCHON(self):
        with pytest.raises(
                SyntaxError,
                match=
            ("SWITCHON is a reserved name, got a Node using it as its name"),
        ):
            loopflow.Node.parse(
                "SWITCHON",
                {},
                multiprocessing.SimpleQueue(),
                multiprocessing.SimpleQueue(),
            )

    def test_fail_when_given_obj_not_dict(self):
        with pytest.raises(SyntaxError,
                           match="Node support parsed by dict only, got 1"):
            loopflow.Node.parse(
                "BAR",
                1,
                multiprocessing.SimpleQueue(),
                multiprocessing.SimpleQueue(),
            )

    def test_fail_when_export_is_not_str_or_list(self):
        with pytest.raises(
                SyntaxError,
                match="export support parsed by only str and list, got 1",
        ):
            loopflow.Node.parse(
                "BAR",
                {"EXPORT": 1},
                multiprocessing.SimpleQueue(),
                multiprocessing.SimpleQueue(),
            )

    def test_fail_when_bad_node_member(self):
        with pytest.raises(SyntaxError, match="Unexpected Node member FOO"):
            loopflow.Node.parse(
                "BAR",
                {"FOO": 1},
                multiprocessing.SimpleQueue(),
                multiprocessing.SimpleQueue(),
            )


class TestFLow(ut.TestCase):

    def test_fail_on_duplicated_node_name(self):
        result_q = multiprocessing.SimpleQueue()
        end_q = multiprocessing.SimpleQueue()
        with pytest.raises(SyntaxError,
                           match="Duplicated node name defined: FOO"):
            loopflow.Flow(
                [
                    loopflow.Node.parse("FOO", {}, result_q, end_q),
                    loopflow.Node.parse("FOO", {}, result_q, end_q),
                ],
                result_q,
                end_q,
            )

    def test_fail_parse_on_list(self):
        with pytest.raises(
                SyntaxError,
                match=r"Flow only support parsed by dict, got \[1\]"):
            loopflow.Flow.parse([1])

    def test_exception_block(self):
        end_q = multiprocessing.SimpleQueue()
        with patch("utensil.loopflow.loopflow.logger.exception"
                  ) as mock, pytest.raises(
                      AttributeError,
                      match="'int' object has no attribute 'values'"):
            flow = loopflow.Flow([], multiprocessing.SimpleQueue(), end_q)
            flow.nodes = 1
            flow.start()
            mock.assert_called_once()
            assert not end_q.empty()


class TestRegisterNodeTasks(ut.TestCase):

    def test_fail_on_duplicated_task_name_registered(self):
        loopflow.reset_node_tasks()

        class MyTask(loopflow.NodeTask):

            def main(self):
                return 1

        loopflow.register_node_tasks(task_map={"FOO": MyTask})
        with pytest.raises(ValueError,
                           match="task name 'FOO' has already been registered"):
            loopflow.register_node_tasks(task_map={"FOO": MyTask})


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
        from utensil.loopflow.loopflow import (Flow, register_node_tasks,
                                               reset_node_tasks)

        from . import simple

        reset_node_tasks()
        register_node_tasks(task_module=simple)
        register_node_tasks(tasks=[basic.GreaterThan])

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
        self.assertEqual(200.5, output)

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
        from utensil.loopflow.loopflow import (Flow, register_node_tasks,
                                               reset_node_tasks)

        reset_node_tasks()
        register_node_tasks(task_module=basic)
        register_node_tasks(task_module=dataflow)

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


class TestForloopFlow(ut.TestCase):

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
        from utensil.loopflow.loopflow import (Flow, register_node_tasks,
                                               reset_node_tasks)

        reset_node_tasks()
        register_node_tasks(task_module=basic)
        register_node_tasks(task_module=dataflow)

        flow_path = os.path.join(FIXTURE_BASE, "forloop.flow")
        flow = Flow.parse_yaml(flow_path)
        results = flow.start()
        logger.debug(results)
        self.assertEqual(1, len(results))
        self.assertEqual(25, results[0][1])


if __name__ == "__main__":
    ut.main()
