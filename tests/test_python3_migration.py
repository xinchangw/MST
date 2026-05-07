import importlib
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"


def _stub_modules():
    numpy_stub = types.ModuleType("numpy")
    pandas_stub = types.ModuleType("pandas")

    joblib_stub = types.ModuleType("joblib")
    joblib_stub.Parallel = object
    joblib_stub.delayed = lambda fn: fn

    leaf_model_mnl_stub = types.ModuleType("leaf_model_mnl")

    class LeafModel(object):
        pass

    leaf_model_mnl_stub.LeafModel = LeafModel
    leaf_model_mnl_stub.get_sub = lambda *args, **kwargs: None
    leaf_model_mnl_stub.are_Ys_diverse = lambda *args, **kwargs: True

    return {
        "numpy": numpy_stub,
        "pandas": pandas_stub,
        "joblib": joblib_stub,
        "leaf_model_mnl": leaf_model_mnl_stub,
    }


@contextmanager
def patched_modules(module_map):
    previous = {name: sys.modules.get(name) for name in module_map}
    try:
        sys.modules.update(module_map)
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


@contextmanager
def temporary_sys_path(entries):
    original = list(sys.path)
    sys.path[:] = [str(entry) for entry in entries]
    try:
        yield
    finally:
        sys.path[:] = original


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


class Python3MigrationTests(unittest.TestCase):
    def test_mst_which_child_multi_handles_unseen_values(self):
        with patched_modules(_stub_modules()):
            mst_module = load_module_from_path("mst_test_module", REPO_ROOT / "mst.py")

        children = mst_module.which_child_multi(["seen", "missing", "missing"], {"seen": 7, "other": 11})
        self.assertEqual(children, [7, 7, 7])

    def test_cart_refit_which_child_multi_handles_unseen_values(self):
        with patched_modules(_stub_modules()):
            cart_module = load_module_from_path(
                "cart_refit_test_module",
                SRC_DIR / "cart_with_mnl_leaf_refitting.py",
            )

        children = cart_module.which_child_multi(["seen", "missing"], {"seen": 3, "fallback": 9})
        self.assertEqual(children, [3, 3])

    def test_src_package_bootstraps_repo_and_src_paths(self):
        with temporary_sys_path([SCRIPTS_DIR]):
            sys.modules.pop("src", None)
            spec = importlib.util.spec_from_file_location(
                "src",
                SRC_DIR / "__init__.py",
                submodule_search_locations=[str(SRC_DIR)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["src"] = module
            try:
                spec.loader.exec_module(module)
                self.assertIn(str(REPO_ROOT), sys.path)
                self.assertIn(str(SRC_DIR), sys.path)
            finally:
                sys.modules.pop("src", None)

    def test_src_mst_wrapper_imports_root_module(self):
        with patched_modules(_stub_modules()):
            with temporary_sys_path([SCRIPTS_DIR]):
                for module_name in ("src", "src.mst", "mst"):
                    sys.modules.pop(module_name, None)
                wrapper = importlib.import_module("src.mst")
                self.assertTrue(hasattr(wrapper, "MST"))
                self.assertEqual(wrapper.MST.__module__, "mst")


if __name__ == "__main__":
    unittest.main()
