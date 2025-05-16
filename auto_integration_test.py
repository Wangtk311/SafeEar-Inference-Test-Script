import unittest
import shutil
import os
import stat
import subprocess
import json
import tempfile
from pathlib import Path
from infer_single_flac import SafeEarInferencer

# 自动化集成测试，涵盖脚本和模块调用多种方式，请保证项目目录有足够的权限和空间(至少剩余3GiB)

def handle_remove_readonly(func, path, exc_info):
    """
    Windows 删除只读或被锁文件时的回调，先修改权限再删除
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"无法删除文件 {path}: {e}")

class TestSafeEarInferencerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 项目根目录
        cls.root = Path.cwd()

    def setUp(self):
        # 为每个测试创建临时目录
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_infer_"))
        # 复制必要文件
        for fname in [
            "conf_infer.yml",
            "audio.flac",
            "Hubert_base_ls960.pt",
            "SpeechTokenizer.pt",
            "model.ckpt",
        ]:
            shutil.copy(self.root / fname, self.test_dir / fname)
        # 切换工作目录
        self.prev_cwd = Path.cwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        # 恢复目录并删除临时目录
        os.chdir(self.prev_cwd)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, onerror=handle_remove_readonly)

    def test_script_default_invocation(self):
        """测试直接运行脚本，不带参数"""
        # 调用脚本并验证结果文件生成
        subprocess.run(
                ["python", str(self.root / "infer_single_flac.py")],
                capture_output=True, text=True, check=True)
        out_file = self.test_dir / "result" / "infer_result.json"
        self.assertTrue(out_file.exists(), "默认脚本调用未生成结果文件")

    def test_script_custom_parameters(self):
        """测试脚本调用时指定 --conf 和 --audio 参数"""
        custom_conf = "custom_conf.yml"
        custom_audio = "test_audio.flac"
        # 复制并重命名
        shutil.copy("conf_infer.yml", custom_conf)
        shutil.copy("audio.flac", custom_audio)
        # 调用脚本并验证结果文件生成
        subprocess.run(
                ["python", str(self.root / "infer_single_flac.py"),
                               "--conf", custom_conf, "--audio", custom_audio],
                capture_output=True, text=True, check=True)
        out_file = self.test_dir / "result" / "infer_result.json"
        self.assertTrue(out_file.exists(), "自定义脚本调用未生成结果文件")

    def test_module_default_invocation(self):
        """测试模块方式调用，使用默认构造参数"""
        inferencer = SafeEarInferencer()
        inferencer.infer()
        out_file = self.test_dir / "result" / "infer_result.json"
        self.assertTrue(out_file.exists(), "模块默认调用未生成结果文件")

    def test_module_custom_invocation(self):
        """测试模块方式调用，使用自定义 conf_path 和 audio_path 参数"""
        inferencer = SafeEarInferencer(
            conf_path="conf_infer.yml", audio_path="audio.flac"
        )
        inferencer.infer()
        out_file = self.test_dir / "result" / "infer_result.json"
        self.assertTrue(out_file.exists(), "模块自定义调用未生成结果文件")

    def test_full_infer_flow(self):
        """验证 infer() 方法完整端到端流程正确性"""
        inferencer = SafeEarInferencer(
            conf_path="conf_infer.yml", audio_path="audio.flac"
        )
        inferencer.infer()
        result_path = Path(inferencer.cfg.output.result_path)
        self.assertTrue(result_path.exists(), "infer() 未生成结果文件")
        data = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertIn("label", data)
        self.assertIn("probs", data)
        self.assertIsInstance(data["label"], list)
        self.assertIsInstance(data["probs"], list)

if __name__ == "__main__":
    unittest.main()
