import unittest
from types import SimpleNamespace

import numpy as np
import torch

from training.smolvla import Adapter


class Tokenizer:
    def __call__(self,text,**kwargs):
        assert all(t.endswith('\n') for t in text)
        return dict(input_ids=torch.ones((len(text),3),dtype=torch.long),
                    attention_mask=torch.ones((len(text),3),dtype=torch.long))


class AdapterContract(unittest.TestCase):
    def test_action_units_round_trip_and_no_diagnostic_inputs(self):
        policy=SimpleNamespace(config=SimpleNamespace(tokenizer_max_length=48),
            model=SimpleNamespace(vlm_with_expert=SimpleNamespace(processor=SimpleNamespace(tokenizer=Tokenizer()))))
        stats={'observation.state':dict(mean=[1.,2.],std=[2.,0.]),
               'action':dict(mean=[.3,.8],std=[.2,.4])}
        adapter=Adapter(policy,stats,'cpu')
        actions=np.array([[.5,.4],[.7,1.2]],np.float32)
        sample={'observation.state':np.array([3.,2.],np.float32),'action':actions,
            'action_is_pad':np.array([False,True]),'task':'move case',
            'observation.images.front':np.zeros((3,256,256),np.float32),
            'observation.images.inspection':np.ones((3,256,256),np.float32),
            'object_qpos':np.ones(7),'source_timestamp':4.}
        batch=adapter.batch([sample])
        self.assertTrue(torch.isfinite(batch['observation.state']).all())
        self.assertTrue(np.allclose(adapter.decode(batch['action'])[0].numpy(),actions))
        self.assertEqual(batch['action_is_pad'].tolist(),[[False,True]])
        self.assertNotIn('object_qpos',batch);self.assertNotIn('source_timestamp',batch)
        inference=adapter.batch([sample],training=False)
        self.assertNotIn('action',inference);self.assertNotIn('action_is_pad',inference)


if __name__=='__main__':unittest.main()
