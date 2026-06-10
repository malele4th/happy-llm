#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from indexing.manifest import compute_index_version


class ManifestTestCase(unittest.TestCase):
    def test_compute_index_version_stable(self) -> None:
        first = compute_index_version()
        second = compute_index_version()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)


if __name__ == "__main__":
    unittest.main()
