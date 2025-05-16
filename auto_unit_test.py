import unittest
import shutil
import os
import stat
import numpy as np
from pathlib import Path
from infer_single_flac import SafeEarInferencer

# 自动化单元测试，请保证项目目录有足够的权限和空间(至少剩余3GiB)
def handle_remove_readonly(func, path, exc_info):
    """
    Windows 删除只读文件或被锁文件时的回调，先修改权限再删除
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"无法删除文件 {path}: {e}")

class TestSafeEarInferencerIntegration(unittest.TestCase):
    def setUp(self):
        # 项目根目录
        self.root = Path.cwd()
        # 创建临时测试目录
        self.test_dir = self.root / "test_infer"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, onerror=handle_remove_readonly)
        self.test_dir.mkdir()
        # 复制配置、音频、模型到测试目录
        for fname in [
            "conf_infer.yml",
            "audio.flac",
            "Hubert_base_ls960.pt",
            "SpeechTokenizer.pt",
            "model.ckpt"
        ]:
            src = self.root / fname
            dst = self.test_dir / fname
            shutil.copy(src, dst)
        # 切换工作目录
        os.chdir(self.test_dir)
        # 初始化 Inferencer
        self.inferencer = SafeEarInferencer(
            conf_path="conf_infer.yml",
            audio_path="audio.flac"
        )

    def tearDown(self):
        # 切回并删除临时目录
        os.chdir(self.root)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, onerror=handle_remove_readonly)

    def test_clean_creates_no_leftover(self):
        # 测试：初始清理不报错，并确实没有残留文件
        self.inferencer._clean()
        self.assertFalse(self.inferencer.feat_dir.exists())
        self.assertFalse(self.inferencer.protocol.exists())
        self.assertFalse(self.inferencer.manifest.exists())

    def test_make_and_clean_protocol_manifest(self):
        # 测试：生成 protocol 和 manifest 文件，然后清理
        self.inferencer._make_single_protocol()
        self.inferencer._make_single_manifest()
        self.assertTrue(self.inferencer.protocol.exists(), "protocol 文件未生成")
        self.assertTrue(self.inferencer.manifest.exists(), "manifest 文件未生成")

        # 清理后应当不存在
        self.inferencer._clean()
        self.assertFalse(self.inferencer.protocol.exists())
        self.assertFalse(self.inferencer.manifest.exists())

    def test_extract_and_save_hubert(self):
        # 测试：保证能从真实 HuBERT 模型提取特征并保存
        self.inferencer._clean()
        self.inferencer._extract_and_save_hubert()
        feat_file = self.inferencer.feat_dir / "audio.npy"
        self.assertTrue(feat_file.exists(), "特征文件未保存")
        arr = np.load(feat_file)
        self.assertIsInstance(arr, np.ndarray, "保存的文件不是 numpy 数组")

if __name__ == "__main__":
    unittest.main()
