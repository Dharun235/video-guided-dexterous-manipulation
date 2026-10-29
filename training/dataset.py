"""Map-style reader for the prepared robot data; PyTorch is not required here.

Returns only camera images, robot proprioception, task text and action targets.
Ground-truth object poses and human-video timing never enter policy inputs.
"""
import json
from pathlib import Path

import cv2
import numpy as np


class RobotDataset:
    def __init__(self,root,split='train',action_horizon=50,include_tactile=False):
        self.root=Path(root);self.split=split;self.action_horizon=action_horizon;self.include_tactile=include_tactile
        if split not in ('train','validation','test'):raise ValueError('Unknown split')
        if action_horizon<1:raise ValueError('action_horizon must be positive')
        self.rows=[json.loads(line) for line in (self.root/'data'/f'{split}.jsonl').read_text().splitlines()]

    def __len__(self):return len(self.rows)

    def action_chunk(self,index):
        row=self.rows[index];actions=[];padding=[];last=np.array(row['action'],dtype=np.float32)
        for offset in range(self.action_horizon):
            j=index+offset
            valid=j<len(self.rows) and self.rows[j]['episode']==row['episode']
            if valid:last=np.array(self.rows[j]['action'],dtype=np.float32)
            actions.append(last.copy());padding.append(not valid)
        return np.array(actions),np.array(padding,dtype=bool)

    def __getitem__(self,index):
        row=self.rows[index]
        result={'observation.state':np.array(row['observation.state'],dtype=np.float32),'task':row['task']}
        if getattr(self,'include_tactile',False):
            tactile=np.asarray(row['observation.tactile'],dtype=np.float32)
            if tactile.shape!=(10,) or not np.isfinite(tactile).all():raise ValueError('Invalid tactile observation')
            result['observation.state']=np.r_[result['observation.state'],tactile]
        for key in ('observation.images.front','observation.images.inspection'):
            bgr=cv2.imread(str(self.root/row[key]))
            if bgr is None:raise FileNotFoundError(row[key])
            result[key]=np.transpose(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(2,0,1)).astype(np.float32)/255.
        result['action'],result['action_is_pad']=self.action_chunk(index)
        return result


class EpisodeDataset(RobotDataset):
    """Read one additional audited training episode using the same contract."""
    def __init__(self,episode,action_horizon=10):
        self.root=Path(episode);self.action_horizon=action_horizon
        if not json.loads((self.root/'audit.json').read_text())['passed']:
            raise ValueError('Additional episode must pass its audit')
        schema=json.loads((self.root/'dataset.json').read_text());x=np.load(self.root/'episode.npz')
        self.rows=[dict(episode='.',task=schema['task'],**{
            'observation.state':x['observation_state'][i].tolist(),'action':x['action'][i].tolist(),
            'observation.images.front':f'images/observation.images.front/{i:06d}.jpg',
            'observation.images.inspection':f'images/observation.images.inspection/{i:06d}.jpg'})
            for i in range(len(x['action']))]
