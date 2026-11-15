"""CPU-only elapsed-rate contracts; no network, credentials or GPU work."""

import unittest
from unittest.mock import Mock, patch

from online_experiments import run_autodl_upload as upload


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Contracts(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.addCleanup(patch.stopall)
        patch.object(upload.time, "monotonic", self.clock.monotonic).start()
        patch.object(upload.time, "sleep", self.clock.sleep).start()
        self.block = upload.transfer.CHUNK_BYTES

    def test_slow_network_writes_are_not_charged_additional_sleep(self):
        throttle = upload.ElapsedRateLimit(32)
        for _ in range(8):
            self.clock.now += 1.0  # Each 8 MiB write already took one second.
            throttle(self.block)
        self.assertEqual(self.clock.sleeps, [])

    def test_fast_writes_are_limited_to_the_requested_rate(self):
        throttle = upload.ElapsedRateLimit(32)
        for _ in range(8):
            throttle(self.block)
        self.assertAlmostEqual(sum(self.clock.sleeps), 2.0)

    def test_actual_write_time_counts_toward_the_elapsed_budget(self):
        throttle = upload.ElapsedRateLimit(32)
        self.clock.now += 0.1
        throttle(self.block)
        self.assertAlmostEqual(self.clock.sleeps[0], 0.15)

    def test_long_idle_cannot_accumulate_an_unbounded_burst(self):
        throttle = upload.ElapsedRateLimit(32)
        self.clock.now += 3600
        throttle(self.block)
        self.assertEqual(self.clock.sleeps, [])
        throttle(self.block)
        self.assertAlmostEqual(self.clock.sleeps[0], 0.25)

    def test_rate_and_byte_validation(self):
        for rate in (0, 0.9, 40.1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                upload.ElapsedRateLimit(rate)
        throttle = upload.ElapsedRateLimit(40)
        for amount in (-1, 0.5, True):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                throttle(amount)

    def test_main_only_substitutes_limiter_and_restores_frozen_helper(self):
        original = upload.transfer.RateLimit
        seen = []
        main = Mock(side_effect=lambda: seen.append(upload.transfer.RateLimit) or 7)
        with patch.object(upload.transfer, "main", main):
            self.assertEqual(upload.main(), 7)
        self.assertEqual(seen, [upload.ElapsedRateLimit])
        self.assertIs(upload.transfer.RateLimit, original)

    def test_limiter_is_restored_even_when_original_main_raises(self):
        original = upload.transfer.RateLimit
        with patch.object(upload.transfer, "main", side_effect=SystemExit(2)), self.assertRaises(SystemExit):
            upload.main()
        self.assertIs(upload.transfer.RateLimit, original)


if __name__ == "__main__":
    unittest.main()
