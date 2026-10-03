from __future__ import annotations

import unittest
import torch

from tools.target_free_torch import CompiledASCII95Torch, V, OFFSETS
from experiments.har_topokv_compiled_v1 import CompiledHARV1, HARIntervention, Op, Stage


def fixture(d: int = 437) -> CompiledASCII95Torch:
    g = torch.Generator().manual_seed(7)
    E = torch.randn(V, d, dtype=torch.float64, generator=g)
    mutual = torch.ones(len(OFFSETS), V, V, dtype=torch.float64)
    values = torch.randn(len(OFFSETS), V, d, dtype=torch.float64, generator=g) * 0.01
    ffn = torch.zeros(V, d, dtype=torch.float64)
    return CompiledASCII95Torch(E, mutual, values, ffn).eval()


class HARCompiledV1Tests(unittest.TestCase):
    def test_observation_equivalence(self):
        model = fixture()
        rels = [(0, 10), (1, 20)]
        base = model.forward_relations(rels)
        wrapped = CompiledHARV1(model).forward_relations(rels)
        self.assertEqual(base["winner_index"], wrapped["winner_index"])
        for key in ("h0", "attention_delta", "hA", "ffn_delta", "hF", "scores"):
            self.assertTrue(torch.equal(base[key], wrapped[key]), key)

    def test_scale(self):
        model = fixture()
        rels = [(0, 10)]
        base = CompiledHARV1(model).forward_relations(rels)
        ctl = CompiledHARV1(model, [HARIntervention(Stage.ATTENTION_DELTA, Op.SCALE, (3,), 2.0)])
        out = ctl.forward_relations(rels)
        self.assertEqual(float(out["attention_delta"][3]), float(base["attention_delta"][3] * 2.0))

    def test_invert(self):
        model = fixture()
        rels = [(0, 10)]
        base = CompiledHARV1(model).forward_relations(rels)
        out = CompiledHARV1(model, [HARIntervention(Stage.POST_ATTENTION, Op.INVERT, (5,))]).forward_relations(rels)
        self.assertEqual(float(out["hA"][5]), float(-base["hA"][5]))

    def test_suppress_not_zero_ablation(self):
        model = fixture()
        rels = [(0, 10)]
        base = CompiledHARV1(model).forward_relations(rels)
        out = CompiledHARV1(model, [HARIntervention(Stage.POST_FFN, Op.SUPPRESS, (8,), 0.25)]).forward_relations(rels)
        self.assertEqual(float(out["hF"][8]), float(base["hF"][8] * 0.25))
        self.assertNotEqual(float(out["hF"][8]), 0.0)

    def test_reroute_conserves_selected_sum(self):
        model = fixture()
        rels = [(0, 10)]
        base = CompiledHARV1(model).forward_relations(rels)
        rule = HARIntervention(Stage.POST_FFN, Op.REROUTE, (1,), 0.4, (2,))
        out = CompiledHARV1(model, [rule]).forward_relations(rels)
        self.assertTrue(torch.allclose(base["hF"][[1,2]].sum(), out["hF"][[1,2]].sum(), rtol=0, atol=1e-12))

    def test_original_model_is_not_mutated(self):
        model = fixture()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        CompiledHARV1(model, [HARIntervention(Stage.POST_FFN, Op.INVERT, (0,1,2))]).predict_next("The dog ")
        after = model.state_dict()
        for k in before:
            self.assertTrue(torch.equal(before[k], after[k]), k)


if __name__ == "__main__":
    unittest.main()
