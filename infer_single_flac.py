import os
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import hydra
from pytorch_lightning import Trainer

from safeear.datas.asvspoof19 import DataModule
from safeear.trainer.safeear_trainer import SafeEarTrainer
from safeear.utils.dump_hubert_feature import HubertFeatureReader


class SafeEarInferencer:
    def __init__(self, conf_path: str = "conf_infer.yml", audio_path: str = "audio.flac"):
        self.conf_path = conf_path
        self.audio_path = Path(audio_path)

        self.feat_dir = Path("feat")
        self.protocol = Path("hubert_protocol.txt")
        self.manifest = Path("hubert_manifest.tsv")
        self.ckpt_path = Path("./Hubert_base_ls960.pt")

        self.cfg = OmegaConf.load(conf_path)
        self.reader = HubertFeatureReader(ckpt_path=str(self.ckpt_path), layer=9, max_chunk=1600000)

    def _extract_and_save_hubert(self):
        feats = self.reader.get_feats(str(self.audio_path))
        feats = feats.transpose(1, 0).cpu().numpy()
        out = self.feat_dir / self.audio_path.name.replace(".flac", ".npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), feats)

    def _make_single_protocol(self):
        uid = self.audio_path.stem
        with open(self.protocol, "w", encoding="utf-8") as f:
            f.write(f"{uid} {uid} bona\n")

    def _make_single_manifest(self):
        root = self.audio_path.parent
        rel = self.audio_path.name
        uid = self.audio_path.stem
        with open(self.manifest, "w", encoding="utf-8") as f:
            f.write(f"{root}\n")
            f.write(f"{rel}\t{uid}\n")

    def _clean(self):
        if self.feat_dir.exists():
            shutil.rmtree(self.feat_dir)
        if self.protocol.exists():
            self.protocol.unlink()
        if self.manifest.exists():
            self.manifest.unlink()

    def infer(self):
        # 1. 清理旧文件
        self._clean()

        # 2. 生成 protocol / manifest
        self._make_single_protocol()
        self._make_single_manifest()

        # 3. 提取 HuBERT 特征
        self._extract_and_save_hubert()

        # 4. 加载 DataModule
        dm: DataModule = hydra.utils.instantiate(self.cfg.datamodule)
        dm.setup(stage="predict")

        # 5. 加载模型
        model: SafeEarTrainer = SafeEarTrainer.load_from_checkpoint(
            self.cfg.ckpt_path,
            decouple_model=hydra.utils.instantiate(self.cfg.decouple_model),
            detect_model=hydra.utils.instantiate(self.cfg.detect_model),
            lr_raw_former=self.cfg.system.lr_raw_former,
            save_score_path=self.cfg.system.save_score_path
        )
        model.eval()

        # 6. 推理
        trainer: Trainer = hydra.utils.instantiate(self.cfg.trainer)
        results = trainer.predict(model, datamodule=dm)

        # 7. 保存结果
        os.makedirs(os.path.dirname(self.cfg.output.result_path), exist_ok=True)
        out = results[0]
        with open(self.cfg.output.result_path, "w", encoding="utf-8") as f:
            json.dump({
                "label": out["label"].tolist(),
                "probs": out["probs"].tolist()
            }, f, ensure_ascii=False, indent=2)

        print(f"推理完成，结果保存在 {self.cfg.output.result_path}")


# 如果仍需命令行执行功能，可保留以下入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="conf_infer.yml", help="推理配置文件路径")
    parser.add_argument("--audio", type=str, default="audio.flac", help="待推理 FLAC 音频路径")
    args = parser.parse_args()

    inferencer = SafeEarInferencer(conf_path=args.conf, audio_path=args.audio)
    inferencer.infer()
