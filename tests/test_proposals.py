from __future__ import annotations

import unittest

from chatuskoti_evals.proposals import ProposalEngine
from chatuskoti_evals.wisdom import WisdomStore


class ProposalTests(unittest.TestCase):
    def test_calibration_mode_uses_adversarial_sequence_first(self) -> None:
        engine = ProposalEngine()
        wisdom = WisdomStore()

        first = engine.propose("vec3", [], wisdom, mode="calibration")
        self.assertEqual(first.name, "pyrrhic_probe")

    def test_challenge_mode_uses_probe_first(self) -> None:
        engine = ProposalEngine()
        wisdom = WisdomStore()

        first = engine.propose("vec3", [], wisdom, mode="challenge")
        self.assertEqual(first.name, "pyrrhic_probe")
