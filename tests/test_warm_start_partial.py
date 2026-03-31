import unittest
from pathlib import Path

import warm_start


class _ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


class WarmStartPartialLoadTest(unittest.TestCase):
    def test_non_strict_warm_start_loads_only_compatible_keys(self):
        loaded = {}

        class FakeModule:
            def state_dict(self):
                return {
                    "encoder.weight": _ShapeOnly((2, 2)),
                    "predictor.weight": _ShapeOnly((4, 4)),
                }

            def load_state_dict(self, state_dict, strict=True):
                loaded["state_dict"] = state_dict
                loaded["strict"] = strict
                return object()

        payload = {
            "encoder.weight": _ShapeOnly((2, 2)),
            "predictor.weight": _ShapeOnly((8, 8)),
            "unused.weight": _ShapeOnly((1,)),
        }

        result = warm_start.apply_warm_start(
            module=FakeModule(),
            checkpoint_path=Path("/tmp/stablewm/pusht/lewm_weights.ckpt"),
            loader=lambda path, map_location=None, weights_only=None: payload,
            strict=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(loaded["state_dict"], {"encoder.weight": payload["encoder.weight"]})
        self.assertFalse(loaded["strict"])


if __name__ == "__main__":
    unittest.main()
