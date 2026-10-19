# 听见：macOS 安装包

下载与你的 Mac 芯片对应的 Actions artifact：`Tingjian-macos-arm64` 适用于 Apple 芯片，`Tingjian-macos-x86_64` 适用于 Intel。解压后将 `Tingjian.app` 移到“应用程序”并双击。应用自带 Python 和运行依赖，页面只监听 `127.0.0.1`；关闭听见的小窗口会停止本地服务。

此公开构建**不包含** `UVR_MDXNET_Main.onnx` 模型、歌曲、录音或历史。未导入模型时，干声音高分析与双通道双人音高分析可用；需要模型的人声分离模式不可用，页面与 API 会显示未就绪。不要把 Windows 的 `.exe` 安装包发给 Mac 用户。

如果你已从合法来源取得与本项目匹配的模型，在应用窗口点击“打开模型文件夹”，将文件命名为 `UVR_MDXNET_Main.onnx` 放入该文件夹，然后重启应用。模型仅存储在当前用户的 `~/Library/Application Support/Tingjian/models/`，不会复制到应用包。程序会校验文件大小和 SHA-256（见 `backend/separation.py`）。本项目不提供模型下载，也未确认该模型权重可再分发。

歌曲、录音与分析数据保存在 `~/Library/Application Support/Tingjian/data/`，与应用包分离。删除应用不会自动删除这些用户数据。

这个 Actions 包仅经过构建 runner 的无界面合成音频测试。它没有 Apple Developer ID 签名或公证，因此 macOS 可能阻止打开；不要关闭系统安全保护来运行来历不明的包。实际麦克风、另一台实体 Mac 上的双击体验与 Gatekeeper 流程尚未测试。

第三方来源见 `THIRD_PARTY_NOTICES.md`。随包的 FFmpeg 版本和编译选项见 `FFMPEG_BUILD.txt`；若该二进制启用了 GPL，包内附 `GPL-3.0.txt`，其对应源码取得方式见 `FFMPEG_SOURCE.txt`。
