import numpy as np
import pytest

from geometry.warp import warp_depth_backward, warp_field_backward, warp_normals_backward


def test_identity_warp_preserves_scalar_and_vector_fields() -> None:
    field = np.arange(20, dtype=np.float32).reshape(4, 5)
    flow = np.zeros((4, 5, 2), dtype=np.float32)
    warped, valid = warp_field_backward(field, flow)
    np.testing.assert_allclose(warped, field)
    assert valid.all()
    vectors = np.zeros((4, 5, 2), dtype=np.float32)
    vectors[..., 0] = 3.0
    vectors[..., 1] = 4.0
    warped_vectors, vector_valid = warp_normals_backward(vectors, flow)
    np.testing.assert_allclose(warped_vectors[..., 0], 0.6)
    np.testing.assert_allclose(warped_vectors[..., 1], 0.8)
    assert vector_valid.all()


def test_translation_warp_gathers_previous_field() -> None:
    field = np.tile(np.arange(5, dtype=np.float32), (4, 1))
    flow = np.zeros((4, 5, 2), dtype=np.float32)
    flow[..., 0] = -1.0
    warped, valid = warp_field_backward(field, flow)
    assert not valid[:, 0].any()
    np.testing.assert_allclose(warped[:, 1:], field[:, :-1])


def test_out_of_frame_and_invalid_source_are_rejected() -> None:
    field = np.ones((3, 4), dtype=np.float32)
    field[1, 1] = np.nan
    flow = np.zeros((3, 4, 2), dtype=np.float32)
    flow[..., 0] = 10.0
    warped, valid = warp_field_backward(field, flow)
    assert not valid.any()
    assert np.isnan(warped).all()
    with pytest.raises(ValueError, match="interpolation"):
        warp_field_backward(field, flow, interpolation="cubic")


def test_source_valid_mask_is_not_mutated() -> None:
    field = np.ones((2, 3), dtype=np.float32)
    field[0, 1] = np.nan
    source_mask = np.ones((2, 3), dtype=bool)
    flow = np.zeros((2, 3, 2), dtype=np.float32)
    warp_field_backward(field, flow, source_mask)
    assert source_mask.all()


def test_subpixel_scalar_warp() -> None:
    """Continuous scalar field with subpixel warp evaluates linear blend accurately."""
    field = np.array([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32)
    flow = np.full((2, 2, 2), -0.5, dtype=np.float32)
    flow[..., 1] = 0.0  # gather from x - 0.5
    warped, valid = warp_field_backward(field, flow)
    # At (y, 1), source is x=0.5 -> blend of 10.0 and 20.0 = 15.0
    assert valid[0, 1]
    assert np.isclose(warped[0, 1], 15.0)


@pytest.mark.parametrize("subpixel_flow", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("scale_pair", [(1.0, 3.0), (0.1, 0.3), (10.0, 30.0)])
def test_subpixel_step_edge_no_phantom_surfaces_mandatory(
    subpixel_flow: float,
    scale_pair: tuple[float, float],
) -> None:
    """Mandatory test: 1->3 step edge must never produce ~2 intermediate depth."""
    fg, bg = scale_pair
    height, width = 8, 16

    depth = np.full((height, width), bg, dtype=np.float32)
    depth[:, :8] = fg  # columns 0..7 foreground, 8..15 background

    # Backward flow shifts sampling by subpixel amount: source_x = x + flow_x
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[..., 0] = subpixel_flow

    warped, valid, warp_conf = warp_depth_backward(depth, flow, discontinuity_threshold=0.15)
    # Interior columns 0..14 have sources within [0, 15]; column 15 source is > 15 (out of frame)
    assert valid[:, :-1].all()
    assert not valid[:, -1].any()

    # Crucial assertion: no phantom intermediate depths anywhere on valid pixels!
    # For fg=1, bg=3: depth must be in {1, 3}, NEVER in (1.2, 2.8)
    intermediate_min = fg + 0.15 * (bg - fg)
    intermediate_max = bg - 0.15 * (bg - fg)

    valid_warped = warped[valid]
    phantom_mask = (valid_warped > intermediate_min) & (valid_warped < intermediate_max)
    assert not phantom_mask.any(), (
        f"Phantom intermediate depth detected! Warped values around boundary: {warped[0, 6:10]}"
    )

    # Every valid pixel must strictly match either foreground or background physical depth layer
    close_to_fg = np.isclose(valid_warped, fg, atol=1e-5 * bg)
    close_to_bg = np.isclose(valid_warped, bg, atol=1e-5 * bg)
    assert (close_to_fg | close_to_bg).all()

    # Near the moving boundary (column 7), confidence should be reduced
    assert warp_conf[:, 7].mean() < 0.5
    # In interior continuous regions, confidence should be high
    assert warp_conf[:, 2].mean() > 0.8
    assert warp_conf[:, 13].mean() > 0.8


def test_warp_depth_backward_validation_and_rejections() -> None:
    depth = np.ones((6, 6), dtype=np.float32)
    flow = np.zeros((6, 6, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="depth must have shape"):
        warp_depth_backward(np.ones((6, 6, 3)), flow)
    with pytest.raises(ValueError, match="discontinuity_threshold"):
        warp_depth_backward(depth, flow, discontinuity_threshold=0.0)
    with pytest.raises(ValueError, match="source_valid_mask"):
        warp_depth_backward(depth, flow, source_valid_mask=np.ones((4, 4), dtype=bool))

    # Out-of-frame coordinates
    flow[..., 0] = 50.0
    warped, valid, conf = warp_depth_backward(depth, flow)
    assert not valid.any()
    assert np.isnan(warped).all()
    assert np.all(conf == 0.0)
