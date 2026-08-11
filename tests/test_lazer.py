from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from osumapper.lazer import open_in_lazer


class LazerTests(unittest.TestCase):
    @patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False)
    @patch("osumapper.lazer.subprocess.Popen")
    @patch("osumapper.lazer.subprocess.run")
    @patch("osumapper.lazer.find_lazer_executable")
    def test_wsl_translates_package_path_for_windows_lazer(
        self,
        find_executable,
        run,
        popen,
    ) -> None:
        executable = Path("/mnt/c/Users/test/AppData/Local/osulazer/current/osu!.exe")
        package = Path("/home/test/osumapper/output/set.osz")
        find_executable.return_value = executable
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=r"\\wsl.localhost\Ubuntu\home\test\osumapper\output\set.osz" + "\n",
        )

        open_in_lazer(package)

        run.assert_called_once_with(
            ["wslpath", "-w", str(package.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
        popen.assert_called_once()
        arguments = popen.call_args.args[0]
        self.assertEqual(arguments[0], str(executable))
        self.assertEqual(
            arguments[1],
            r"\\wsl.localhost\Ubuntu\home\test\osumapper\output\set.osz",
        )


if __name__ == "__main__":
    unittest.main()
