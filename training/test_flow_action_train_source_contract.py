"""Pure-source guards for memory-sensitive formal DDP configuration."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("flow_action_train.py")


class FlowActionTrainSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))

    def test_ddp_gradients_are_bucket_views(self) -> None:
        accelerators = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Accelerator"
        ]
        self.assertEqual(len(accelerators), 1, "expected one formal Accelerator")
        accelerator_keywords = {
            item.arg: item.value for item in accelerators[0].keywords
        }
        kwargs_handlers = accelerator_keywords.get("kwargs_handlers")
        self.assertIsInstance(kwargs_handlers, ast.List)
        handlers = [
            node
            for node in kwargs_handlers.elts
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DistributedDataParallelKwargs"
        ]
        self.assertEqual(len(handlers), 1, "expected one formal DDP kwargs handler")
        keywords = {item.arg: item.value for item in handlers[0].keywords}
        value = keywords.get("gradient_as_bucket_view")
        self.assertIsInstance(value, ast.Constant)
        self.assertIs(value.value, True)

    def test_unconcatenated_video_features_are_released(self) -> None:
        deleted_names = {
            target.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Delete)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("feats", deleted_names)

    def test_formal_vram_ceiling_is_63_gib(self) -> None:
        assignments = {
            target.id: node.value
            for node in self.tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        ceiling = assignments.get("FORMAL_MAX_VRAM_GIB")
        self.assertIsInstance(ceiling, ast.Constant)
        self.assertEqual(ceiling.value, 63.0)

    def test_zro_is_built_after_accelerator_initializes_distributed(self) -> None:
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        accelerator_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "Accelerator"
        ]
        zro_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "ZeroRedundancyOptimizer"
        ]
        self.assertEqual(len(accelerator_calls), 1)
        self.assertEqual(len(zro_calls), 2)
        adapter_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "DtypeShardedAdamW"
        ]
        self.assertEqual(len(adapter_calls), 1)
        self.assertLess(accelerator_calls[0].lineno, adapter_calls[0].lineno)

        for zro_call in zro_calls:
            zro_keywords = {item.arg: item.value for item in zro_call.keywords}
            optimizer_class = zro_keywords.get("optimizer_class")
            self.assertIsInstance(optimizer_class, ast.Attribute)
            self.assertEqual(optimizer_class.attr, "AdamW")
            overlap = zro_keywords.get("overlap_with_ddp")
            self.assertIsInstance(overlap, ast.Constant)
            self.assertIs(overlap.value, False)

    def test_zro_parameters_are_partitioned_by_dense_type(self) -> None:
        adapter = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DtypeShardedAdamW"
        )
        init = next(
            node
            for node in adapter.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertIsNotNone(init)
        source = ast.unparse(init)
        self.assertIn("torch.typename(parameter)", source)
        self.assertIn("for dense_type in sorted(by_dense_type)", source)

    def test_only_empty_startup_state_is_serializable(self) -> None:
        adapter = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DtypeShardedAdamW"
        )
        state_method = next(
            node
            for node in adapter.body
            if isinstance(node, ast.FunctionDef) and node.name == "state_dict"
        )
        source = ast.unparse(state_method)
        self.assertIn("self._step_has_run", source)
        self.assertIn("raise RuntimeError", source)
        self.assertIn("super().state_dict()", source)

    def test_hybrid_optimizer_is_real_adamw_plus_dtype_safe_zro(self) -> None:
        adapter = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DtypeHybridAdamW"
        )
        source = ast.unparse(adapter)
        self.assertIn("torch.optim.AdamW(replicated_params", source)
        self.assertIn("torch.typename(parameter)", source)
        self.assertIn("ZeroRedundancyOptimizer", source)
        self.assertNotIn("torch.empty", source)
        self.assertNotIn("torch.zeros", source)
        self.assertIn("self._step_has_run", source)


if __name__ == "__main__":
    unittest.main()
