from types import SimpleNamespace

import src.models.vllm_video as vllm_video_module
from src.models import load_model
from src.models.vllm_video import VLLMVideoModel, _video_path_to_file_uri


def test_load_model_routes_vllm_video_prefix_to_native_video_backend():
    model = load_model("vllm_video/qwen3_vl_32b_inst")

    assert isinstance(model, VLLMVideoModel)
    assert model.model_id == "qwen3_vl_32b_inst"
    assert model.cache_namespace == "vllm_video_qwen3_vl_32b_inst__vanilla"


def test_vllm_video_build_content_uses_video_url_file_uri():
    model = VLLMVideoModel(model_id="qwen3_vl_32b_inst")

    content = model._build_content("What is happening?", "file:///tmp/clip.mp4")

    assert content[0]["type"] == "text"
    assert "Question: What is happening?" in content[0]["text"]
    assert content[1] == {
        "type": "video_url",
        "video_url": {"url": "file:///tmp/clip.mp4"},
        "uuid": "file:///tmp/clip.mp4",
    }


def test_vllm_video_path_to_file_uri_validates_with_pyav(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video bytes")
    opened_paths = []

    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        @property
        def streams(self):
            return SimpleNamespace(video=[object()])

    def fake_open(path):
        opened_paths.append(path)
        return FakeContainer()

    monkeypatch.setattr(vllm_video_module.av, "open", fake_open)

    assert _video_path_to_file_uri(str(video_path)) == video_path.resolve().as_uri()
    assert opened_paths == [str(video_path.resolve())]
