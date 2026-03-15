from pathlib import Path

import hydra

from run_sample import run_sampling


@hydra.main(version_base=None, config_path="../configs", config_name="sample")
def main(cfg) -> None:
    run_sampling(cfg, project_root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
