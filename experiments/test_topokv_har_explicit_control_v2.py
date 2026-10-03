#!/usr/bin/env python3
"""Tests for representation-addressable TopoKV/HAR integration."""
import unittest
import torch
from tools.target_free_torch import CompiledASCII95Torch
from experiments.topokv_har_explicit_control_v2 import (
    ExplicitTopoKVHAR,HARController,HAROp,RepresentationControl,RepresentationIndex,Site
)

class TestExplicitControl(unittest.TestCase):
    def fixture(self):
        d=437
        g=torch.Generator().manual_seed(19)
        E=torch.randn(95,d,generator=g,dtype=torch.float64)
        mutual=torch.ones(14,95,95,dtype=torch.float64)
        av=torch.randn(14,95,d,generator=g,dtype=torch.float64)*0.01
        ffn=torch.randn(95,d,generator=g,dtype=torch.float64)*0.01
        model=CompiledASCII95Torch(E,mutual,av,ffn).eval()
        names=[f"REP.{i:03d}" for i in range(d)]
        return model,names

    def test_observation_is_bit_exact(self):
        model,names=self.fixture()
        rel=[(6,1),(5,2),(4,3)]
        base=model.forward_relations(rel)
        got=ExplicitTopoKVHAR(model,names).forward_relations(rel)
        for k in ("h0","attention_delta","hA","ffn_delta","hF","scores"):
            self.assertTrue(torch.equal(base[k],got[k]),k)
        self.assertEqual(base["winner_index"],got["winner_index"])

    def test_named_amplification(self):
        model,names=self.fixture()
        rel=[(6,1)]
        base=ExplicitTopoKVHAR(model,names).forward_relations(rel)
        reps=RepresentationIndex(names)
        har=HARController(reps,[RepresentationControl(Site.ATTENTION,HAROp.AMPLIFY,(names[7],),2.0)])
        got=ExplicitTopoKVHAR(model,names,har).forward_relations(rel)
        self.assertEqual(got["attention_delta"][7],base["attention_delta"][7]*2)

    def test_named_suppression_is_not_zero_ablation(self):
        model,names=self.fixture()
        rel=[(6,1)]
        base=ExplicitTopoKVHAR(model,names).forward_relations(rel)
        reps=RepresentationIndex(names)
        har=HARController(reps,[RepresentationControl(Site.FFN,HAROp.SUPPRESS,(names[11],),0.25)])
        got=ExplicitTopoKVHAR(model,names,har).forward_relations(rel)
        self.assertEqual(got["ffn_delta"][11],base["ffn_delta"][11]*0.25)
        if base["ffn_delta"][11] != 0:
            self.assertNotEqual(got["ffn_delta"][11],0)

    def test_named_inversion(self):
        model,names=self.fixture()
        rel=[(6,1)]
        base=ExplicitTopoKVHAR(model,names).forward_relations(rel)
        reps=RepresentationIndex(names)
        har=HARController(reps,[RepresentationControl(Site.RESIDUAL_POST_ATTENTION,HAROp.INVERT,(names[13],))])
        got=ExplicitTopoKVHAR(model,names,har).forward_relations(rel)
        self.assertEqual(got["hA"][13],-base["hA"][13])

    def test_named_reroute_conserves_pair_sum(self):
        model,names=self.fixture()
        rel=[(6,1)]
        base=ExplicitTopoKVHAR(model,names).forward_relations(rel)
        reps=RepresentationIndex(names)
        har=HARController(reps,[RepresentationControl(
            Site.RESIDUAL_POST_FFN,HAROp.REROUTE,(names[17],),0.4,(names[18],)
        )])
        got=ExplicitTopoKVHAR(model,names,har).forward_relations(rel)
        self.assertTrue(torch.allclose(got["hF"][17]+got["hF"][18],base["hF"][17]+base["hF"][18]))

    def test_model_state_unchanged(self):
        model,names=self.fixture()
        before={k:v.clone() for k,v in model.state_dict().items()}
        reps=RepresentationIndex(names)
        har=HARController(reps,[RepresentationControl(Site.ATTENTION,HAROp.INVERT,(names[3],))])
        ctl=ExplicitTopoKVHAR(model,names,har)
        ctl.forward_relations([(6,1)])
        ctl.assert_weights_unchanged(before)

    def test_unknown_representation_rejected(self):
        model,names=self.fixture()
        reps=RepresentationIndex(names)
        har=HARController(reps,[RepresentationControl(Site.ATTENTION,HAROp.AMPLIFY,("NOT.A.REP",),2)])
        with self.assertRaises(KeyError):
            ExplicitTopoKVHAR(model,names,har).forward_relations([(6,1)])

if __name__=="__main__":
    unittest.main()
