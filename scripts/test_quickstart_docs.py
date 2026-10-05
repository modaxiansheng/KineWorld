"""CPU documentation contracts; never downloads or runs model commands."""

import ast
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "track1"))
sys.path.insert(0, str(REPO))
import infer_track1
import precompute_action_flow
import validate_track1
from scripts import prepare_inference_episode, prepare_robotwin_manifest


class QuickstartDocsTests(unittest.TestCase):
    def test_raft_template_matches_documented_download_directory(self):
        guide = (REPO / "README_zh-CN.md").read_text(encoding="utf-8")
        config = (REPO / "configs/train_public.example.env").read_text(encoding="utf-8")
        path = re.search(r"^export KINEWORLD_RAFT_WEIGHTS_PATH=(.+)$", config, re.M)[1]
        self.assertIn("-o " + path, guide)
        self.assertIn("$KINEWORLD_ROOT/" + path, guide)

    def test_local_links_and_documented_python_arguments(self):
        text = (REPO / "README_zh-CN.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("```") % 2, 0)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("https://", "http://", "#")):
                self.assertTrue((REPO / target.split("#")[0]).exists(), target)
        checked = []
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            for line in block.replace("\\\n", " ").splitlines():
                if not line.strip().startswith("python "):
                    continue
                argv = shlex.split(line)
                if argv[1] == "-c":
                    ast.parse(argv[2])
                    continue
                if not argv[1].endswith(".py"):
                    continue
                script, options = argv[1], argv[2:]
                with patch.object(sys, "argv", [script, *options]):
                    if script == "track1/infer_track1.py":
                        args = infer_track1.parse_args()
                        infer_track1.validate_args(args)
                        self.assertEqual(args.input_profile, "custom")
                    elif script == "track1/validate_track1.py":
                        self.assertEqual(validate_track1.parse_args().mode, "outputs")
                    elif script == "track1/precompute_action_flow.py":
                        precompute_action_flow.build_parser().parse_args(options)
                    elif script == "scripts/prepare_inference_episode.py":
                        prepare_inference_episode.build_parser().parse_args(options)
                    elif script == "scripts/prepare_robotwin_manifest.py":
                        prepare_robotwin_manifest.build_parser().parse_args(options)
                    elif script == "script/render_robot_only.py":
                        self.assertEqual(options, ["place_dual_shoes", "aloha-agilex_clean_50"])
                        self.assertTrue((REPO / "data_generation" / script).is_file())
                    else:
                        self.fail(f"Unvalidated documented script: {script}")
                checked.append(script)
        self.assertEqual(len(checked), 8)

    @unittest.skipUnless(os.environ.get("KINEWORLD_TEST_BASH"), "set KINEWORLD_TEST_BASH for shell syntax checks")
    def test_every_bash_block_parses_without_execution(self):
        text = (REPO / "README_zh-CN.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
        self.assertGreater(len(blocks), 15)
        for index, block in enumerate(blocks):
            result = subprocess.run([os.environ["KINEWORLD_TEST_BASH"], "-n"],
                                    input=block, text=True, encoding="utf-8", capture_output=True)
            self.assertEqual(result.returncode, 0, f"block {index}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
