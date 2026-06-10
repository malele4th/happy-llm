#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from parsing.rules import heading_max_len, load_parser_rules


class ParserRulesTestCase(unittest.TestCase):
    def test_heading_max_len_positive(self) -> None:
        self.assertGreater(heading_max_len(), 0)

    def test_load_parser_rules_has_keys(self) -> None:
        rules = load_parser_rules()
        self.assertIn("heading_max_len", rules)


if __name__ == "__main__":
    unittest.main()
