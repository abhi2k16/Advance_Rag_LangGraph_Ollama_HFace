import unittest
from pathlib import Path

import main


class ProjectStructureTests(unittest.TestCase):
    def test_workflow_modules_exist(self) -> None:
        for workflow, exists in main.available_workflows():
            self.assertTrue(exists, workflow.module)
            self.assertTrue(main.module_path(workflow.module).is_file())

    def test_helper_modules_exist(self) -> None:
        for module, _ in main.UTILITY_MODULES:
            self.assertTrue(main.module_path(module).is_file(), module)

    def test_package_init_files_exist(self) -> None:
        root = Path(__file__).resolve().parent
        self.assertTrue((root / "langchain_impl" / "__init__.py").is_file())
        self.assertTrue((root / "langgraph_impl" / "__init__.py").is_file())

    def test_workflow_alias_resolution(self) -> None:
        self.assertEqual(main.resolve_workflow("langgraph").key, "memory")
        self.assertEqual(main.resolve_workflow("generation").key, "advanced")
        self.assertEqual(main.resolve_workflow("index").key, "indexing")


if __name__ == "__main__":
    unittest.main()
