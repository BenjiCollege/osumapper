from __future__ import annotations

import io
import sys
import unittest
from unittest.mock import patch

from osumapper.cli import _use_utf8_output


class CliOutputEncodingTests(unittest.TestCase):
    def test_star_ratings_survive_a_non_utf8_console(self) -> None:
        # Windows hands a redirected stream the ANSI code page, where U+2605
        # is unencodable. Progress lines and difficulty names both contain it,
        # so without this a full-set run aborts after the first tier message.
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
            with self.assertRaises(UnicodeEncodeError):
                stdout.write("Expert+ 7.00★")
                stdout.flush()

            _use_utf8_output()
            sys.stdout.write("Expert+ 7.00★")
            sys.stdout.flush()

            self.assertEqual(sys.stdout.encoding, "utf-8")
            self.assertEqual(sys.stderr.encoding, "utf-8")

    def test_streams_without_reconfigure_are_left_alone(self) -> None:
        # Captured or wrapped streams need not expose reconfigure; the CLI must
        # keep working rather than fail at startup.
        class Bare:
            encoding = "ascii"

        bare = Bare()
        with patch.object(sys, "stdout", bare), patch.object(sys, "stderr", bare):
            _use_utf8_output()

            self.assertEqual(bare.encoding, "ascii")


if __name__ == "__main__":
    unittest.main()
