# 第三方来源与署名

本仓库仅收录此项目的源码和合成测试代码。以下上游软件通过用户确认后的安装脚本取得，模型权重和第三方二进制均未收入仓库。版本以 `requirements.txt` 和 `requirements-dev.txt` 为准。

## 算法、参考实现与模型

| 内容 | 本项目使用方式 | 已核实的来源与授权信息 |
|---|---|---|
| YIN 基频估计 | `backend/pitch.py` 的独立实现 | Alain de Cheveigné、Hideki Kawahara，2002 年论文 [“YIN, a fundamental frequency estimator for speech and music”](https://pubmed.ncbi.nlm.nih.gov/12002874/)；引用论文不等于复制论文代码。 |
| Ultimate Vocal Remover（UVR）及 MDX-Net 推理流程 | 分离参数和协议的参考；未收入 UVR GUI 源文件 | [UVR README 的 License 与 Credits](https://github.com/Anjok07/ultimatevocalremovergui#license)声明其 GUI 代码为 MIT，并要求第三方使用其模型时署名 UVR 与开发者。感谢 Anjok07、aufr33；README 也将原始 MDX-Net AI 代码归功于 Kuielab 和 Woosung Choi。README 指向的独立 `LICENSE` 文件在 2026-10-07 核查时为 404，因此此处如实记录上游声明，不将其扩展为本仓库全部内容的许可。 |
| `UVR_MDXNET_Main.onnx` | 用户明确确认后由 `setup.ps1` 从上游下载到本机 `models/` | [TRvlvr 的公开 UVR 模型发布页](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)；预期大小 66,759,214 字节，SHA-256 `8289784cda38543ff431add4070662813311a8cccfc0112ca82f76d9dba2b4ca`。UVR README 对第三方模型使用提出 MIT 署名要求，但本次未在模型下载仓库找到针对该 ONNX 文件的独立许可文本。**本仓库不再分发权重；如需重新分发模型，先取得明确授权。** |
| NCM 容器格式 | `backend/ncm.py` 按格式重新实现读取；未收入上游源文件或可执行程序 | 参考 [taurusxin/ncmdump](https://github.com/taurusxin/ncmdump)。仓库标为 MIT，见其 [LICENSE.txt](https://github.com/taurusxin/ncmdump/blob/main/LICENSE.txt)；该许可文本的版权行仍为模板占位符。感谢 ncmdump 与更早格式研究者。格式兼容不授予任何商业歌曲的使用或传播权。 |

若未来加入上游源码片段、模型文件或打包的第三方二进制，应重新核对适用许可证并随分发物保留相应完整许可与版权声明。当前表述不声称这些内容归本项目所有。

## 运行时依赖

| 组件 | 来源与许可提示 |
|---|---|
| [FastAPI](https://github.com/fastapi/fastapi)、[Uvicorn](https://github.com/encode/uvicorn) | 分别按上游 MIT、BSD-3-Clause 许可安装；不复制其源码到仓库。 |
| [NumPy](https://github.com/numpy/numpy)、[SciPy](https://github.com/scipy/scipy) | BSD 风格许可证；由用户安装到本地环境。 |
| [python-soundfile](https://github.com/bastibe/python-soundfile) | Python 包为 BSD-3-Clause；其音频库 [libsndfile](https://libsndfile.github.io/libsndfile/) 为 LGPL-2.1 或 LGPL-3 可选，平台 wheel 可能包含该库。 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | Microsoft MIT 许可；本项目默认使用 CPU provider。 |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE) 与 [FFmpeg](https://ffmpeg.org/legal.html) | Python 包为 BSD-2-Clause。其 wheel 可能带 FFmpeg 可执行文件；本机所用构建启用了 GPLv3 选项。源码仓库不包含该二进制；如另行打包安装程序，必须重新审查 FFmpeg 的许可与源码提供义务。 |
| [pytest](https://github.com/pytest-dev/pytest)、[httpx](https://github.com/encode/httpx)、[Playwright Python](https://github.com/microsoft/playwright-python) | 仅测试依赖，分别按上游 MIT、BSD-3-Clause、Apache-2.0 许可安装。 |

以上为来源和分发边界说明，不代替各上游项目的完整许可文件；安装的 wheel 应保留它们各自附带的许可证。
