# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from examples.offline_inference.minimax_h3 import end2end as example
from vllm_omni.model_executor.models.minimax_h3.timeline_guides import GUIDES_EXTRA_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_help_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", str(Path(example.__file__)), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "--guide" in result.stdout
    assert "Negative pixel-frame indices" in result.stdout


def test_ordered_local_guides_keep_negative_indices_and_repeated_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Descriptor validation checks files and budgets, not media decoding.
    for name in ("middle, x=y.png", "tail.mp4", "sound.flac"):
        (tmp_path / name).write_bytes(b"nonempty local source")
    args = example.parse_args(
        [
            "--guide",
            "36",
            "image=middle, x=y.png",
            "--guide",
            "-22",
            "video=tail.mp4",
            "audio=sound.flac",
            "--guide",
            "0",
            "audio=~/sound.flac",
            "--guide",
            "36",
            "image=middle, x=y.png",
            "--guide",
            "1",
            "image=middle, x=y.png",
            "audio=sound.flac",
            "--guide",
            "5",
            "video=tail.mp4",
        ]
    )
    image = str(tmp_path / "middle, x=y.png")
    video = str(tmp_path / "tail.mp4")
    audio = str(tmp_path / "sound.flac")
    assert example._parse_timeline_guides(args.guide) == [
        {"frame_index": 36, "image": image},
        {"frame_index": -22, "video": video, "audio": audio},
        {"frame_index": 0, "audio": audio},
        {"frame_index": 36, "image": image},
        {"frame_index": 1, "image": image, "audio": audio},
        {"frame_index": 5, "video": video},
    ]
    assert example._resolve_task_and_mm_data(args) == (None, {})


@pytest.mark.parametrize("entries", [None, []])
def test_no_guides(entries):
    assert example._parse_timeline_guides(entries) == []
    assert example.parse_args([]).guide is None


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([["36"]], "expected FRAME_INDEX"),
        ([["1.5", "image=file.png"]], "FRAME_INDEX must be an integer"),
        ([["true", "image=file.png"]], "FRAME_INDEX must be an integer"),
        ([["0", "image="]], "expected image=PATH"),
        ([["0", "image"]], "expected image=PATH"),
        ([["0", "mask=file.png"]], "expected image=PATH"),
        ([["0", "image=a.png", "image=b.png"]], "duplicate image"),
        ([["0", "audio=a.wav", "audio=b.wav"]], "duplicate audio"),
        ([["0", "image=a.png", "video=b.mp4"]], "cannot combine image with video"),
        ([["0", "video=https://example.com/clip.mp4"]], "trusted local file path"),
        ([["0", "image=missing.png"]], "cannot access guide image file"),
        ([["0", "audio=file.wav"]] * 9, "too many timeline guide entries"),
    ],
)
def test_invalid_guide_inputs(entries, message):
    with pytest.raises(ValueError, match=message):
        example._parse_timeline_guides(entries)


@pytest.mark.parametrize("directory", [False, True])
def test_guides_require_nonempty_regular_files(tmp_path, directory):
    source = tmp_path / "source"
    if directory:
        source.mkdir()
    else:
        source.touch()
    with pytest.raises(ValueError, match="nonempty regular file"):
        example._parse_timeline_guides([["0", f"audio={source}"]])


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("guided", [False, True])
def test_main_forwards_guides_only_in_sampling_extras(tmp_path, monkeypatch, task, guided):
    source = tmp_path / "source.png"
    source.write_bytes(b"nonempty local source")
    argv = [
        "end2end.py",
        "--model",
        "local-model",
        "--task",
        task,
        "--output",
        str(tmp_path / "out"),
        "--quality",
        "lossless",
        "--num-frames",
        "124",
        "--seed",
        "42",
        "--prompts",
        "First prompt",
        "Second prompt",
    ]
    expected_mm = {}
    if task == "fl2va":
        argv += ["--image-path", str(source)]
        expected_mm = {"image": str(source)}
    elif task == "ref2va":
        argv += ["--video-path", str(source)]
        expected_mm = {"video": [str(source)]}
    if guided:
        argv += ["--guide", "-1", f"image={source}", "--guide", "36", f"image={source}"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    closed = []

    class FakeOmni:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(images=[np.zeros((1, 2, 2, 3), dtype=np.uint8)], multimodal_output={})]

        def close(self):
            closed.append(True)

    # Exercise CLI-to-request wiring without loading a model or inference stack.
    for name, attributes in {
        "vllm_omni.entrypoints.omni": {"Omni": FakeOmni},
        "vllm_omni.diffusion.data": {"DiffusionParallelConfig": SimpleNamespace},
        "vllm_omni.inputs.data": {"OmniDiffusionSamplingParams": SimpleNamespace},
        "vllm_omni.diffusion.utils.media_utils": {"mux_video_audio_bytes": lambda *a, **kw: b"muxed"},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    example.main()
    assert len(calls) == 1
    assert closed == [True]
    params = calls[0]["sampling_params_list"]
    assert params.seed == 42 and params.num_frames == 124 and params.quality == "lossless"
    expected_extra = {"aspect_ratio": "16:9", "task": task}
    if guided:
        expected_extra[GUIDES_EXTRA_KEY] = [
            {"frame_index": -1, "image": str(source)},
            {"frame_index": 36, "image": str(source)},
        ]
    assert params.extra_args == expected_extra
    assert calls[0]["prompts"] == [
        {"prompt": prompt, **({"multi_modal_data": expected_mm} if expected_mm else {})}
        for prompt in ("First prompt", "Second prompt")
    ]
    assert source.exists()  # Offline sources remain owned by the user.
