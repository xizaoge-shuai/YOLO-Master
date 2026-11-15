"""Upload through the frozen transport with bounded, elapsed-time rate control.

The original uploader owns authentication, manifests, resume verification and
publication. This adapter only avoids charging sleep for time already spent
writing through the network. Idle credit is capped at one 8 MiB source block.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import transfer_autodl_resume as transfer


class ElapsedRateLimit:
    """Account cumulative bytes against elapsed time, with bounded idle credit."""

    def __init__(self, mib):
        if not math.isfinite(mib) or not 1 <= mib <= 40:
            raise ValueError("rate-mib must be 1..40")
        self.rate = mib * 1024**2
        self.started = time.monotonic()
        self.amount = 0
        self.burst = transfer.CHUNK_BYTES

    def __call__(self, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError("Rate accounting requires a nonnegative integer byte count")
        tick = time.monotonic()
        # Discard idle credit beyond one block; slow writes still pay no sleep.
        prior_deadline = self.started + self.amount / self.rate
        if tick - prior_deadline > self.burst / self.rate:
            self.started = tick - (self.amount + self.burst) / self.rate
        self.amount += amount
        delay = self.started + self.amount / self.rate - tick
        if delay > 0:
            time.sleep(delay)


def main():
    original = transfer.RateLimit
    transfer.RateLimit = ElapsedRateLimit
    try:
        return transfer.main()
    finally:
        transfer.RateLimit = original


if __name__ == "__main__":
    raise SystemExit(main())
