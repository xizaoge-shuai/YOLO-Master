"""CPU contracts for an isolated RGB residual on the cached detector's P3."""

import importlib
import importlib.util
import io

import pytest

torch = pytest.importorskip("torch")
detector_module = pytest.importorskip("scripts.d1_train_cached_detector")
CachedLatentDetector = detector_module.CachedLatentDetector


@pytest.fixture(autouse=True)
def one_cpu_thread():
    """Avoid thread-pool overhead for the small CPU models in this suite."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(mode="rgb", seed=37):
    """Create the proposed model API, with a useful failure before implementation."""
    name = "online_experiments.detail_probe_model"
    assert importlib.util.find_spec(name) is not None, "DetailDetector module has not been implemented"
    cls = importlib.import_module(name).DetailDetector
    torch.manual_seed(seed)
    return cls(in_channels=(8, 8, 8), nc=3, imgsz=64, epochs=2, detail_mode=mode)


def make_raw(dtype=torch.float32):
    generator = torch.Generator().manual_seed(131)
    features = tuple(torch.randn(2, 8, 4, 4, generator=generator, dtype=dtype) for _ in range(3))
    rgb = torch.rand(2, 3, 64, 64, generator=generator, dtype=dtype)
    return features, rgb


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            assert_nested_equal(first, second)
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["none", "rgb", "lowpass"])
def test_initial_parent_state_and_global_rng_are_unchanged(mode):
    torch.manual_seed(37)
    reference = CachedLatentDetector(in_channels=(8, 8, 8), nc=3, imgsz=64, epochs=2)
    next_reference = torch.rand(8)
    candidate = make_model(mode)
    next_candidate = torch.rand(8)
    assert torch.equal(next_reference, next_candidate)
    state = candidate.state_dict()
    for key, value in reference.state_dict().items():
        # LatentMixture also serializes dictionary-valued routing extra state.
        assert_nested_equal(value, state[key])
    if mode == "none":
        assert state.keys() == reference.state_dict().keys()
        assert candidate.detail_branch is None
        assert candidate.detail_parameter_count == 0


@pytest.mark.parametrize("mode", ["none", "rgb", "lowpass"])
@pytest.mark.parametrize("fusion", ["router_only", "mean_value", "deepest_value"])
def test_initial_outputs_equal_parent_in_both_inference_paths(mode, fusion):
    from online_experiments.fusion_probe import configure_fusion

    torch.manual_seed(37)
    reference = CachedLatentDetector(in_channels=(8, 8, 8), nc=3, imgsz=64, epochs=2).eval()
    candidate = make_model(mode).eval()
    configure_fusion(reference, fusion)
    configure_fusion(candidate, fusion)
    features, rgb = make_raw()
    raw = features if mode == "none" else (*features, rgb)
    with torch.no_grad():
        assert_nested_equal(reference.pyramid(features), candidate.pyramid(raw))
        assert_nested_equal(reference(features), candidate(raw))
        assert_nested_equal(reference.training_predictions(features), candidate.training_predictions(raw))


def test_rgb_and_lowpass_have_identical_parameterization():
    rgb = make_model("rgb")
    lowpass = make_model("lowpass")
    assert_nested_equal(rgb.state_dict(), lowpass.state_dict())
    assert rgb.detail_parameter_count == lowpass.detail_parameter_count
    assert 0 < rgb.detail_parameter_count < 10_000
    final = rgb.detail_branch[-1]
    assert final.in_channels == 32 and final.out_channels == 64
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0


@pytest.mark.parametrize("mode", ["rgb", "lowpass"])
def test_nonzero_branch_changes_only_p3(mode):
    model = make_model(mode).eval()
    features, rgb = make_raw()
    with torch.no_grad():
        reference = CachedLatentDetector.pyramid(model, features)
        model.detail_branch[-1].bias.fill_(0.25)
        actual = model.pyramid((*features, rgb))
    assert torch.allclose(actual[0], reference[0] + 0.25)
    assert torch.equal(actual[1], reference[1])
    assert torch.equal(actual[2], reference[2])


@pytest.mark.parametrize("mode", ["rgb", "lowpass"])
def test_branch_receives_detection_gradients_after_zero_init_step(mode):
    model = make_model(mode).train()
    features, rgb = make_raw()
    optimizer = torch.optim.SGD(model.detail_branch.parameters(), lr=0.1)
    first_conv = model.detail_branch[0]
    final_conv = model.detail_branch[-1]
    for step in range(2):
        model.zero_grad(set_to_none=True)
        predictions = model.training_predictions((*features, rgb))["one2many"]
        loss = predictions["scores"].square().mean() + predictions["boxes"].square().mean()
        loss.backward()
        assert final_conv.weight.grad is not None
        assert torch.isfinite(final_conv.weight.grad).all()
        assert torch.count_nonzero(final_conv.weight.grad) > 0
        if step == 0:
            assert torch.count_nonzero(first_conv.weight.grad) == 0
        else:
            for parameter in model.detail_branch.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                assert torch.count_nonzero(parameter.grad) > 0
        optimizer.step()


def test_lowpass_removes_checkerboard_but_preserves_constant_and_normalization():
    rgb_model = make_model("rgb")
    lowpass_model = make_model("lowpass")
    row = torch.arange(64)
    checkerboard = ((row[:, None] + row[None, :]) % 2).float()[None, None].expand(2, 3, -1, -1)
    raw_prepared = rgb_model.prepare_detail_input(checkerboard)
    low_prepared = lowpass_model.prepare_detail_input(checkerboard)
    assert torch.equal(raw_prepared, (checkerboard - 114.0 / 255.0) * 2.0)
    assert raw_prepared.std() > 0.9
    assert low_prepared.std() < 1e-6
    expected = torch.full_like(checkerboard, (0.5 - 114.0 / 255.0) * 2.0)
    assert torch.allclose(low_prepared, expected, atol=1e-6, rtol=0)
    padding = torch.full_like(checkerboard, 114.0 / 255.0)
    # Area pooling accumulates 256 float32 samples before centering; allow its
    # observed ~1.4e-6 rounding error without changing the specified operation order.
    assert torch.allclose(lowpass_model.prepare_detail_input(padding), torch.zeros_like(padding), atol=2e-6, rtol=0)


@pytest.mark.parametrize("mode", ["rgb", "lowpass"])
def test_output_dtype_device_and_dimensions_follow_parent(mode):
    model = make_model(mode).double().eval()
    features, rgb = make_raw(dtype=torch.float64)
    with torch.no_grad():
        output = model.pyramid((*features, rgb))
    for tensor, shape in zip(output, [(2, 64, 8, 8), (2, 128, 4, 4), (2, 256, 2, 2)]):
        assert tensor.shape == shape
        assert tensor.dtype == torch.float64
        assert tensor.device == rgb.device
    prepared = model.prepare_detail_input(rgb)
    assert prepared.dtype == rgb.dtype and prepared.device == rgb.device and prepared.shape == rgb.shape


@pytest.mark.parametrize("mode", ["none", "rgb", "lowpass"])
def test_state_dict_roundtrip_retains_predictions(mode):
    model = make_model(mode).eval()
    if mode != "none":
        with torch.no_grad():
            model.detail_branch[-1].weight.normal_(std=0.01)
    features, rgb = make_raw()
    raw = features if mode == "none" else (*features, rgb)
    serialized = io.BytesIO()
    torch.save(model.state_dict(), serialized)
    serialized.seek(0)
    restored = make_model(mode, seed=91).eval()
    restored.load_state_dict(torch.load(serialized, weights_only=True))
    with torch.no_grad():
        assert_nested_equal(model(raw), restored(raw))


def test_rejects_unknown_detail_mode():
    with pytest.raises(ValueError, match="detail_mode"):
        make_model("blur")


@pytest.mark.parametrize("mode,count", [("none", 4), ("rgb", 3), ("lowpass", 3)])
def test_rejects_wrong_raw_arity(mode, count):
    model = make_model(mode)
    features, rgb = make_raw()
    with pytest.raises(ValueError, match="cached levels"):
        model.pyramid((*features, rgb)[:count])


@pytest.mark.parametrize("bad_shape", [(2, 1, 64, 64), (2, 3, 32, 32), (2, 3, 64)])
def test_rejects_malformed_rgb_shape(bad_shape):
    model = make_model()
    features, _ = make_raw()
    with pytest.raises(ValueError, match="RGB"):
        model.pyramid((*features, torch.zeros(bad_shape)))


def test_rejects_integer_rgb_and_batch_mismatch():
    model = make_model().eval()
    features, rgb = make_raw()
    with pytest.raises(ValueError, match="floating"):
        model.pyramid((*features, rgb.to(torch.uint8)))
    with pytest.raises(ValueError, match="P3"):
        model.pyramid((*features, rgb[:1]))
