import os
import wandb
def set_wandb(config):
    # set environment variables
    os.environ["WANDB_BASE_URL"] = "https://toyota.wandb.io" # トヨタのwandb
    os.environ["WANDB_PROJECT"] = config.competition  # プロジェクト名
    # os.environ["WANDB_DISABLE_GPU"] = "true"
    # os.environ["WANDB_DISABLE_CODE"] = "true"
    # os.environ["WANDB_DISABLE_STATS"] = "true"
    # os.environ["WANDB_SILENT"] = "true"
    # get wandb_api_key (for TOYOTA in house environment)
    wandb_api_key = os.getenv("WANDB_API_KEY") # export wandb_api_key before running program
    if not wandb_api_key:
        raise EnvironmentError("WANDB_API_KEY is not set in the environment. Please set it before running the script.")
    os.environ["WANDB_API_KEY"] = wandb_api_key

    # その他の設定
    os.environ["NCCL_DEBUG"] = "INFO"  # NCCLのデバッグ情報を出力
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用するGPUを指定
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"  # トークナイザーの並列実行を無効化

    # WandBの初期化（必要であれば）
    wandb.login()