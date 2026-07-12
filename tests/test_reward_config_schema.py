from omegaconf import OmegaConf

from teleboost.reward.config_schema import normalize_reward_config


def test_vlm_reward_normalization_does_not_disable_mm_processor_cache():
    cfg = OmegaConf.create(
        {
            "reward": {
                "reward_model": {
                    "enable": True,
                    "adapter": "video_vlm",
                    "rollout": {
                        "engine_kwargs": {
                            "vllm": {
                                "allowed_local_media_path": "/tmp",
                            }
                        }
                    },
                }
            }
        }
    )

    normalize_reward_config(cfg)

    vllm_kwargs = cfg.reward.reward_model.rollout.engine_kwargs.vllm
    assert cfg.reward.reward_model.enable is True
    assert "mm_processor_cache_gb" not in vllm_kwargs
