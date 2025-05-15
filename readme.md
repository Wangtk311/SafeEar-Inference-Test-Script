# SafeEar推理脚本

弄了好长时间，终于把SafeEar的推理脚本搞出来了，经过在ASVSPOOF2019数据集上的测试，应该能够得到和标签一样的结果。下面是使用说明。

## 1.结构说明

```
推理时可以替换的部分：
audio.flac：要推理的音频文件，仅支持flac格式
model.ckpt：训练好的SafeEar模型检查点

推理时无需修改的部分：
infer_single_flac.py：上层推理脚本，已写好无需修改
conf_infer.yml：推理配置文件，与训练配置文件类似，已写好无需修改
hubert_manifest.tsv：HuBERT特征化的manifest文件，包含了音频文件的路径和对应的标签，脚本会自动生成无需修改
hubert_protocol.txt：HuBERT特征化的protocol文件，包含了音频文件的路径和对应的标签，脚本会自动生成无需修改
Hubert_base_ls960.pt：HuBERT模型权重，已下载好无需修改
SpeechTokenizer.pt：音频Tokenizer权重，已下载好无需修改

推理结果：
result/infer_result.json：推理结果文件，包含分类结果和对应的置信度
```

## 2.推理步骤

该模块已经对整个推理过程进行了封装，因此可以提供脚本运行+模块调用两种推理方式，十分方便。

- 数据及环境准备
    - 整体下载所有文件并放置到一个文件夹，命名为TestInference。
    - 模型运行的虚拟环境为SafeEar官方提供的conda虚拟环境。
    - 将要推理的flac音频文件保存到与`infer_single_flac.py`同级的目录下，命名为`audio.flac`。
    - 将要使用的预训练SafeEar模型检查点ckpt文件保存到与`infer_single_flac.py`同级的目录下，命名为`model.ckpt`。
    - 由于仓库大小限制，对于三个模型文件，可能无法上传至GitHub仓库，我不想配置LFS。因此，我已将其上传至网盘，您可以通过网盘链接将它们下载，并放置到`TestInference`文件夹下，即根目录中。对于`model.ckpt`，推荐使用您自己训练的模型检查点（目前我们的版本可能性能并不好）。三个文件的下载百度网盘链接：https://pan.baidu.com/s/14lrsrdz8R-Vy3PjNr67MGg?pwd=sfer 提取码: sfer

- 运行推理脚本
    - 命令行方式调用：
        ```bash
        # 按照默认方式直接调用（推荐）。
        (safeear) python infer_single_flac.py

        # 如果想指定推理配置或者修改推理的音频位置和名称，也可以采用以下方式（不推荐，因为推理器类的实现有可能存在潜在问题，或者推理配置可能存在潜在问题，这部分没有经过测试，故使用时应保持谨慎）。
        (safeear) python infer_single_flac.py --conf my_config.yml --audio my_audio.flac
        ```
    - 模块调用：
        ```python
        # 在TestInference文件夹外的脚本中调用。注意flac文件必须格式正确（ASVSPOOF2019数据集中的flac音频格式是正确的）。
        from TestInference.infer_single_flac import SafeEarInferencer

        # 按照默认方式直接调用（推荐）。
        inferencer = SafeEarInferencer()
        inferencer.infer()

        # 如果想指定推理配置或者修改推理的音频位置和名称，也可以采用以下方式（不推荐，因为推理器类的实现有可能存在潜在问题，或者推理配置可能存在潜在问题，这部分没有经过测试，故使用时应保持谨慎）。
        inferencer = SafeEarInferencer(conf_path="my_config.yml",audio_path="my_audio.flac")
        inferencer.infer()
        ```
- 运行完成后，推理结果会保存在`result/infer_result.json`文件中，包含分类结果和对应的置信度，可以直接解析。

## 3.推理结果解析

推理结果文件`result/infer_result.json`举例如下：
```json
{
  "label": [ 0 ],
  "probs": [ [ 0.9792123436927795, 0.0207876767963171 ] ]
}
```
- label：推理结果分类。
- probs：推理结果置信度，包含二分类的置信度值(和为1.0)。

> **声明：仅做参考和实验，可能存在很多问题，不保证该脚本的正确性，请慎用！**


# SafeEar Inference Script

After considerable effort, the SafeEar inference script has been completed. After testing on the ASVSPOOF2019 dataset, the same result as the label is achievable. Below is the usage guide.

## 1. Structure Overview

```
Files you can replace during inference:
audio.flac # The audio file to be inferred (only supports FLAC format)
model.ckpt # The trained SafeEar model checkpoint

Files unecessary to modify:
infer_single_flac.py # Top-level inference script, already implemented
conf_infer.yml # Inference configuration file, similar to the training config
hubert_manifest.tsv # Manifest for HuBERT feature extraction (auto-generated)
hubert_protocol.txt # Protocol file for HuBERT features (auto-generated)
Hubert_base_ls960.pt # HuBERT model weights (pre-downloaded)
SpeechTokenizer.pt # Audio tokenizer weights (pre-downloaded)

Inference result:
result/infer_result.json # Inference output containing class label and confidence scores
```

## 2. Inference Steps

This module supports both **script-based** and **module-based** inference, making it flexible and convenient.

- Data & Environment Preparation
  - Download all files and put them in a folder named `TestInference`.
  - Use the conda environment provided by SafeEar.
  - Save the audio file you want to infer (`.flac` format) in the same directory as `infer_single_flac.py` and name it `audio.flac`.
  - Save the trained SafeEar checkpoint file as `model.ckpt` in the same directory.
  - Note: For `model.ckpt`, it is recommended to use your own trained checkpoint. Due to GitHub's size limit, model files cannot be hosted directly(I donnot want to setup LFS). Therefore, they have been uploaded to a cloud drive:Download model files (BaiduNetDisk) https://pan.baidu.com/s/14lrsrdz8R-Vy3PjNr67MGg?pwd=sfer Extract code: `sfer`.


- Run the Inference Script

  - Command-line usage:
    ```bash
    # Use the default setup (recommanded)
    (safeear) python infer_single_flac.py

    # Optional: Custom config and audio (not recommended because of possible code errors)
    (safeear) python infer_single_flac.py --conf my_config.yml --audio my_audio.flac
    ```
  - Python module usage:
    ```Python
    # Call the inference in another script (outside TestInference folder)
    from TestInference.infer_single_flac import SafeEarInferencer

    # Recommended usage
    inferencer = SafeEarInferencer()
    inferencer.infer()

    # Optional: Specify config and audio path manually (not recommended because of possible code errors)
    inferencer = SafeEarInferencer(conf_path="my_config.yml", audio_path="my_audio.flac")
    inferencer.infer()
    ```

- After execution, results will be saved to `result/infer_result.json`, which contains classification results and their corresponding confidence scores.

## 3. Inference Result Format

Example `result/infer_result.json`:

```Json
{
  "label": [ 0 ],
  "probs": [ [ 0.9792123436927795, 0.0207876767963171 ] ]
}
```

- `label`: The predicted class index.

- `probs`: Confidence scores for each class (binary classification; values sum to 1.0).

> **This script is provided for only reference and experimental use only. It may contain bugs or issues. Please use it with caution and do not assume its correctness!**
