# Copyright 2025-2026 TeleAI and the TeleBoost contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from unittest import TestCase

from unit_tests.test_utils import spawn

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def success(rank, world_size, q):
    logging.info(f"hello rank{rank}")
    q.put("True")


def fail(rank, world_size, q):
    q.put("False")


def crash(rank, world_size, q):
    raise ValueError(f"intentional child failure on rank {rank}")


class testMPTesting(TestCase):
    def testSuccess(self):
        size = 4
        q = spawn(size, success)

        cnt = 0
        while not q.empty():
            res = q.get()
            self.assertEqual(res, "True")
            cnt += 1
        self.assertEqual(cnt, size)

    def testFail(self):
        size = 4
        q = spawn(size, fail)

        cnt = 0
        while not q.empty():
            res = q.get()
            self.assertEqual(res, "False")
            cnt += 1
        self.assertEqual(cnt, size)

    def testChildFailurePropagates(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"rank 0 exitcode=1, rank 1 exitcode=1",
        ):
            spawn(2, crash)
