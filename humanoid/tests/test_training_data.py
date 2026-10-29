import json

import numpy as np

from training.dataset import RobotDataset


def test_action_chunks_do_not_cross_episode_boundaries(tmp_path):
    (tmp_path/'data').mkdir()
    rows=[{'episode':'a','action':[1.,2.]},{'episode':'a','action':[3.,4.]},
          {'episode':'b','action':[100.,200.]}]
    (tmp_path/'data/train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    dataset=RobotDataset(tmp_path,action_horizon=4)
    actions,padding=dataset.action_chunk(1)
    assert np.array_equal(actions,np.array([[3.,4.]]*4))
    assert padding.tolist()==[False,True,True,True]
