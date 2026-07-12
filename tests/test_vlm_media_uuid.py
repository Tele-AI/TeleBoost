import aiohttp  # noqa: F401 - dependency contract for loaded reward modules
from teleboost.reward.adapters import video_vlm_score


def test_video_vlm_message_sets_top_level_media_uuid():
    messages = video_vlm_score._build_messages(
        "data:video/mp4;base64,ZmFrZQ==",
        "a dog running",
        "teleboost-video-abc",
    )

    video_part = messages[0]["content"][0]
    assert video_part["type"] == "video_url"
    assert video_part["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert video_part["uuid"] == "teleboost-video-abc"
