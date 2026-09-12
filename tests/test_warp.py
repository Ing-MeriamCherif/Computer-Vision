import numpy as np

from geometry.warp import warp_field_backward, warp_normals_backward


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
    with np.testing.assert_raises(ValueError):
        warp_field_backward(field, flow, interpolation="cubic")


def test_source_valid_mask_is_not_mutated() -> None:
    field = np.ones((2, 3), dtype=np.float32)
    field[0, 1] = np.nan
    source_mask = np.ones((2, 3), dtype=bool)
    flow = np.zeros((2, 3, 2), dtype=np.float32)
    warp_field_backward(field, flow, source_mask)
    assert source_mask.all()
