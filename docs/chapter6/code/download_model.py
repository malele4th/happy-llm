import subprocess
import sys

# 勿在代码里写死 HF_ENDPOINT=https://hf-mirror.com：
# 新版 huggingface_hub 用 HEAD 拉元数据时不跟随跨域 308；
# hf-mirror 会跳到 huggingface.co，导致缺少 X-Repo-Commit 等头并报错。
# 能直连 huggingface.co 时不要设镜像；
# 国内受限网络请用可直出文件的镜像、或 hf-mirror 的 hfd 脚本。

subprocess.run(
    [
        sys.executable,
        "-m",
        "huggingface_hub.cli.hf",
        "download",
        "Qwen/Qwen2.5-1.5B",
        "--local-dir",
        "autodl_model/qwen-1.5b",
    ],
    check=True,
)
