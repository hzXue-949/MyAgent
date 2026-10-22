import importlib
import os
import pkgutil
import unittest
from unittest.mock import patch

import examples
import my_agent
from my_agent.paths import ENV_FILE, PROJECT_ROOT


class ProjectLayoutTests(unittest.TestCase):
    def test_modules_import_without_network_or_model_initialization(self):
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv"), patch("socket.socket.connect", side_effect=AssertionError("导入时禁止访问网络")), patch("fastembed.TextEmbedding", side_effect=AssertionError("导入时不应加载模型")), patch("openai.OpenAI", side_effect=AssertionError("导入时不应创建模型客户端")):
            for package in (my_agent, examples):
                for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                    with self.subTest(module=module.name):
                        importlib.import_module(module.name)

    def test_env_file_is_anchored_to_project_root(self):
        self.assertEqual(ENV_FILE, PROJECT_ROOT / ".env")

    def test_reorganized_packages_exist(self):
        for name in ("core", "agents", "llm", "tools", "memory"):
            self.assertTrue((PROJECT_ROOT / "my_agent" / name / "__init__.py").is_file())
