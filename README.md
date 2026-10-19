# 唱歌音高实验室（本机版）

macOS 用户：请查看 [听见 macOS 安装说明](README_MAC.md)。公开构建包含 Apple 芯片和 Intel 两种架构的 `.app`，不包含许可未明确的模型权重。

本项目提供中文 Web 界面和仅监听 `127.0.0.1` 的本地处理服务。可上传参考歌曲与演唱录音，离线分离混音中的人声和伴奏，查看音高、音名与音分偏差；也可用麦克风检测干声，在播放已分析的参考曲时进行实时比对。混音不会被直接当成单声部人声评分。戴耳机可以减少伴奏串入麦克风。

所有录音和缓存留在本机。仓库不包含用户歌曲、录音、分离结果、分析数据库、模型权重或第三方可执行文件。`scripts/generate_fixtures.py` 和测试中的声音由算法合成，不是商业歌曲。仅处理你有权使用的音频。

## 安装与启动

已在 Windows x64、Python 3.13 上验证。首次运行请在 PowerShell 中执行：

```powershell
.\setup.ps1
.\start.ps1
```

也可双击 `start.cmd` 启动。浏览器打开 <http://127.0.0.1:8765>。`setup.ps1` 会先列出依赖和模型来源，要求输入 `YES` 后才在项目 `.venv` 内安装依赖并从上游下载模型；不会全局安装或使用付费服务。安装了 Python 的其他版本时，请先按脚本提示准备 Python 3.13。若端口已占用，可用 `.\start.ps1 -Port 8766`。

分离模型为上游发布的 `UVR_MDXNET_Main.onnx`，下载源为 [TRvlvr 的 UVR 模型发布页](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)，预期 SHA-256 为 `8289784cda38543ff431add4070662813311a8cccfc0112ca82f76d9dba2b4ca`。模型约 66.8 MB，**不随仓库分发**。UVR 的使用说明与模型署名要求见[第三方来源说明](THIRD_PARTY_NOTICES.md)；权重自身的独立再分发许可尚未核实，因此不要把它打入发行包。

## 使用边界

- 混音先由 UVR MDX-Net 模型离线分离，随后才分析参考人声。麦克风的实时模式检测用户干声，不做实时混音分离。
- 音高估计基于 [de Cheveigné 与 Kawahara 的 YIN 论文](https://pubmed.ncbi.nlm.nih.gov/12002874/)。低音量、低置信度、多人合唱和分离残留处不应当被视为可靠音准评分。
- 参考曲与录音可以调整时间偏移和转调。已分析歌曲保存在本机原唱库，可在刷新和重启后恢复；相同文件和模式的再次上传会复用分析。
- 新结果以高精度 FLAC 保存。旧 WAV 缓存仍可读取；软件不会自动删除旧歌曲。
- NCM 导入仅限用户自行提供并有权处理的文件。格式实现参考 [ncmdump](https://github.com/taurusxin/ncmdump)，仓库未包含其程序或商业音频。

## 测试

测试依赖需要在首次安装时选择 `.\setup.ps1 -WithTests`。随后可运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pitch tests.test_separation -v
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_ncm.py tests/test_storage.py -q -p no:cacheprovider
node tests/frontend.test.mjs
```

`tests/ui_smoke.py` 使用本机 Chrome 和合成麦克风流；真实硬件麦克风需要使用者现场测试。测试生成物留在 `evidence/` 与 `.test-runs/`，已被忽略。`scripts/build.py` 只生成源码包，不应放入私人音频。

## 开源与授权

本项目原创源码采用 [MIT 许可证](LICENSE)。第三方组件与模型各自适用其条款，详见 [第三方来源说明](THIRD_PARTY_NOTICES.md)；模型权重不随仓库分发。
