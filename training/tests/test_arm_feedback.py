import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from humanoid.arm_feedback import ArmFeedback
from training.arm_feedback import Residual


class ArmFeedbackContract(unittest.TestCase):
    def test_export_matches_torch_and_gain_zero_is_identity(self):
        torch.manual_seed(7);rng=np.random.default_rng(7);model=Residual()
        with torch.no_grad():model.layers[-1].weight.normal_(0,.1)
        mean=rng.normal(size=45).astype(np.float32);std=rng.uniform(.1,2,45).astype(np.float32)
        cap=np.full(7,.12,np.float32);weights=dict(mean=mean,std=std,cap=cap)
        for i,layer in enumerate(model.layers):
            weights[f'w{i}']=layer.weight.detach().numpy();weights[f'b{i}']=layer.bias.detach().numpy()
        feature=rng.normal(size=45).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'feedback.npz';np.savez(path,**weights)
            result=ArmFeedback(path).correction(feature[:19],feature[19:32],feature[32:])
            with torch.no_grad():expected=cap*model(torch.tensor(np.clip((feature-mean)/std,-8,8))).numpy()
            self.assertTrue(np.allclose(result,expected,atol=1e-6))
            self.assertTrue(np.all(np.abs(result)<=.12))
            high=ArmFeedback(path,2).correction(feature[:19],feature[19:32],feature[32:])
            self.assertTrue(np.all(np.abs(high)<=.12))
            self.assertEqual(result.shape,(7,))
            self.assertTrue(np.array_equal(ArmFeedback(path,0).correction(feature[:19],feature[19:32],feature[32:]),np.zeros(7)))
            with self.assertRaises(ValueError):ArmFeedback(path).correction(np.zeros(26),feature[19:32],feature[32:])


if __name__=='__main__':unittest.main()
