#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from utils import normalize_report_date, parse_date_filter, parse_quarter_from_path, resolve_date_filter


class UtilsTestCase(unittest.TestCase):
    def test_normalize_report_date(self) -> None:
        self.assertEqual(normalize_report_date("20251211"), "20251211")
        self.assertEqual(normalize_report_date("2025-12-11"), "20251211")
        self.assertEqual(normalize_report_date("invalid"), "")

    def test_parse_date_filter(self) -> None:
        self.assertEqual(parse_date_filter("2025年12月catchii"), (2025, 12))
        self.assertEqual(parse_date_filter("2025年进展"), (2025, None))

    def test_parse_quarter_from_path(self) -> None:
        path = "2025/Q2/工作周报-20250605.docx"
        self.assertEqual(parse_quarter_from_path(path), "2025Q2")

    def test_resolve_date_filter(self) -> None:
        self.assertEqual(resolve_date_filter("2025年12月", None, None, True), (2025, 12))
        self.assertEqual(resolve_date_filter("2025年12月", 2024, 1, True), (2024, 1))
        self.assertEqual(resolve_date_filter("2025年12月", None, None, False), (None, None))


if __name__ == "__main__":
    unittest.main()
