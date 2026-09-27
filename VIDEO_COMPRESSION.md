# 视频对应说明

## 视觉识别演示

- 原文件：`04_Evidence/Physical_Video/acceptance_20260907_192601.avi`
- 本包文件：`04_Evidence/Physical_Video/acceptance_20260907_192601.mp4`
- 原文件SHA256：`40beefc96132b33a8725bd16b0eb7393709a7f662f250d011eea165928b72ff8`
- 原尺寸1280×720，30帧/秒，5513帧，约183.77秒。
- `04_Evidence/RawRuns/v4验收原始目录/`中的同名AVI与上述原文件完全相同，本包不再重复存放。

## 六格流程录像

- 路径：`04_Evidence/RawRuns/六格完整流程_20260908_032515/实验3_完整运行_20260908_032515.mp4`
- 本包保留同一路径，改用H.264编码。
- 原尺寸960×540，30帧/秒，约206.30秒。

两段视频均使用H.264、CRF 20重新编码，没有裁剪画面、删帧、改变播放速度或修改识别标注。两段原视频均不含音轨。压缩属于有损编码，原始文件仍保存在同级的 `Group15_Experiment3_Submission_20260922` 完整版中。

其余视频、代码、模型、配置、训练资料和原始日志未改动。原始日志中的视频名称、哈希和历史路径仍指向原始素材；`05_Records/physical_results.json` 的 `file` 与 `sha256` 对应包内视频，`original_file` 与 `original_sha256` 保留原始AVI身份。包内全部文件校验值见 `SHA256SUMS.txt`。
