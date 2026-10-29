"""Prevent apparent generalization caused by trajectory/split leakage."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from humanoid.book_collection_audit import inspect


def collection(tmp_path):
    entries=[]
    for group,split in enumerate(('train','train','validation','validation','test','test')):
        relative=f'{split}/group_{group:02d}/episode_00';p=tmp_path/relative
        (p/'images/observation.images.front').mkdir(parents=True)
        image=np.full((256,256,3),group*30,np.uint8)
        cv2.imwrite(str(p/'images/observation.images.front/000000.jpg'),image)
        np.savez_compressed(p/'episode.npz',example=np.array([group]))
        image_hash=hashlib.sha256(image.tobytes()).hexdigest()
        trajectory_hash=hashlib.sha256((p/'episode.npz').read_bytes()).hexdigest()
        (p/'audit.json').write_text(json.dumps(dict(passed=True,frames=1,image_sha256=image_hash,episode_sha256=trajectory_hash)))
        (p/'domain.json').write_text(json.dumps(dict(seed=100+group)))
        (p/'dataset.json').write_text(json.dumps(dict(state_joint_names=['example_joint'],action_actuator_names=['example_motor'])))
        (p/'report.json').write_text(json.dumps(dict(checks={'test_task':True},goal_position_m=[0,0,.013],task_success=True)))
        entries.append(dict(episode=relative,physical_group=group,split=split,verified=True))
    (tmp_path/'index.json').write_text(json.dumps(dict(episodes=entries,groups=6,variants_per_group=1)))
    return entries


def test_separate_groups_are_eligible(tmp_path):
    collection(tmp_path);inspect(tmp_path)
    assert json.loads((tmp_path/'collection-audit.json').read_text())['passed']


def test_shared_movement_cannot_cross_split(tmp_path):
    entries=collection(tmp_path);entries[-1]['physical_group']=0
    (tmp_path/'index.json').write_text(json.dumps(dict(episodes=entries,groups=6,variants_per_group=1)))
    with pytest.raises(SystemExit):inspect(tmp_path)
    checks=json.loads((tmp_path/'collection-audit.json').read_text())['checks']
    assert not checks['group_split_isolation']


def test_identical_images_cannot_cross_split(tmp_path):
    entries=collection(tmp_path)
    first=json.loads((tmp_path/entries[0]['episode']/'audit.json').read_text())
    p=tmp_path/entries[-1]['episode']/'audit.json';last=json.loads(p.read_text())
    last['image_sha256']=first['image_sha256'];p.write_text(json.dumps(last))
    with pytest.raises(SystemExit):inspect(tmp_path)
    assert not json.loads((tmp_path/'collection-audit.json').read_text())['checks']['no_identical_image_sequences_across_splits']
