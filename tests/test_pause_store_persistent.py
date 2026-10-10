from __future__ import annotations

import os
from pathlib import Path

from people.pause_store import copy_checkpoint, put_checkpoint, take_checkpoint


def test_pause_store_memory_roundtrip(tmp_path: Path, monkeypatch) -> None:
    # 默认不设置 PEOPLE_CHECKPOINT_DIR，使用纯内存模式
    monkeypatch.delenv("PEOPLE_CHECKPOINT_DIR", raising=False)
    obj = {"x": 1, "y": [1, 2, 3]}
    tok = put_checkpoint(obj)
    snap = copy_checkpoint(tok)
    assert snap == obj
    taken = take_checkpoint(tok)
    assert taken == obj
    assert take_checkpoint(tok) is None


def test_pause_store_disk_roundtrip(tmp_path: Path, monkeypatch) -> None:
    # 配置 PEOPLE_CHECKPOINT_DIR，验证磁盘持久化可用
    d = tmp_path / "checkpoints"
    monkeypatch.setenv("PEOPLE_CHECKPOINT_DIR", str(d))
    obj = {"foo": "bar", "nested": {"a": 1}}
    tok = put_checkpoint(obj)
    # 磁盘文件应该存在
    pkl = next(iter(d.glob(f"{tok}.pkl")), None)
    assert pkl is not None
    # copy 不消费
    cp = copy_checkpoint(tok)
    assert cp == obj
    assert pkl.is_file()
    # take 应消费（文件被删除），内容相同
    taken = take_checkpoint(tok)
    assert taken == obj
    assert not pkl.exists()
    # 再次获取返回 None
    assert take_checkpoint(tok) is None
