from pathlib import Path

import hydra

from run_train import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg) -> None:
    run_training(cfg, project_root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
