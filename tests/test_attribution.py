from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from osumapper import __original_author__, __original_project__
from osumapper.cli import main


class AttributionTests(unittest.TestCase):
    def test_original_project_is_credited_by_package_and_cli(self) -> None:
        self.assertEqual(__original_author__, "kotritrona")
        self.assertEqual(__original_project__, "https://github.com/kotritrona/osumapper")

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["credits"])

        self.assertEqual(result, 0)
        self.assertIn("Original creator: kotritrona", output.getvalue())
        self.assertIn(__original_project__, output.getvalue())


if __name__ == "__main__":
    unittest.main()
