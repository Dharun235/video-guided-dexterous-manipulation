import json

import cv2
import numpy as np

from training.dataset import RobotDataset


def test_tactile_is_opt_in_and_diagnostic_fields_never_enter_inputs(tmp_path):
    (tmp_path/'data').mkdir();cv2.imwrite(str(tmp_path/'image.jpg'),np.zeros((256,256,3),np.uint8))
    row={'episode':'e0','task':'place case','observation.state':list(range(19)),
        'observation.tactile':list(range(10)),'action':[.2]*13,'action.reference':[.8]*13,
        'object_qpos':[7]*7,'source_timestamp':6.,'supervision.phase':'carry',
        'observation.images.front':'image.jpg','observation.images.inspection':'image.jpg'}
    (tmp_path/'data/train.jsonl').write_text(json.dumps(row)+'\n')
    plain=RobotDataset(tmp_path,action_horizon=2)[0]
    sample=RobotDataset(tmp_path,action_horizon=2,include_tactile=True)[0]
    assert plain['observation.state'].shape==(19,)
    assert sample['observation.state'].shape==(29,)
    assert np.array_equal(sample['observation.state'][19:],np.arange(10))
    assert np.allclose(sample['action'],.2)
    assert sample['action_is_pad'].tolist()==[False,True]
    assert not {'object_qpos','source_timestamp','supervision.phase','action.reference'}&sample.keys()
