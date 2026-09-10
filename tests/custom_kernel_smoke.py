import unittest

import mlx.core as mx


class CustomKernelSmoke(unittest.TestCase):
    def call(self, kernel, inputs, shape, dtype, *, grid=None, threadgroup=None, **kwargs):
        size = 1
        for dimension in shape:
            size *= dimension
        return kernel(
            inputs=inputs,
            output_shapes=[shape],
            output_dtypes=[dtype],
            grid=grid or (size, 1, 1),
            threadgroup=threadgroup or (min(size, 32), 1, 1),
            stream=mx.gpu,
            **kwargs,
        )[0]

    def test_msl_body_runs_on_gpu(self):
        kernel = mx.fast.metal_kernel(
            name="omarchy_affine",
            input_names=["values", "scale"],
            output_names=["out"],
            source="""
                uint index = thread_position_in_grid.x;
                out[index] = values[index] * scale + 3.0f;
            """,
        )
        values = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        out = self.call(kernel, [values, 2.0], values.shape, values.dtype)
        self.assertEqual(out.tolist(), [5.0, 7.0, 9.0, 11.0])

    def test_templates_scalars_bfloat_and_multiple_outputs(self):
        kernel = mx.fast.metal_kernel(
            name="omarchy_arguments",
            input_names=["a", "b", "c", "d"],
            output_names=["out1", "out2"],
            source="""
                uint elem = thread_position_in_grid.x;
                T tmp = a[0];
                if (enabled) {
                    out1[elem] = a[1] + b[2] + c[3] + d + extra;
                } else {
                    out1[elem] = tmp;
                }
                out2[elem] = a[1] + b[2] + c[1] - d;
            """,
        )
        out1, out2 = kernel(
            inputs=[
                mx.array([1.0, 2.0]),
                mx.array([3, 4, 5]),
                mx.array([6.0, 7.0, 8.0, 9.0], dtype=mx.bfloat16),
                2.0,
            ],
            template=[("enabled", True), ("extra", 3), ("T", mx.float16)],
            grid=(4, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(4,), (4,)],
            output_dtypes=[mx.float32, mx.int32],
            stream=mx.gpu,
        )
        self.assertEqual(out1.tolist(), [21.0] * 4)
        self.assertEqual(out2.tolist(), [12] * 4)

    def test_noncontiguous_shape_and_stride_metadata(self):
        values = mx.arange(12, dtype=mx.float32).reshape(3, 4).T
        kernel = mx.fast.metal_kernel(
            name="omarchy_strides",
            input_names=["inp"],
            output_names=["out"],
            ensure_row_contiguous=False,
            source="""
                uint elem = thread_position_in_grid.x;
                uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
                out[elem] = inp[loc];
            """,
        )
        out = self.call(kernel, [values], values.shape, values.dtype)
        self.assertEqual(out.tolist(), values.tolist())

    def test_header_helper_and_threadgroup_attribute(self):
        helper = mx.fast.metal_kernel(
            name="omarchy_helper",
            input_names=["values"],
            output_names=["out"],
            header="""
                template <typename T>
                T twice(T value) { return value + value; }
            """,
            source="""
                uint elem = thread_position_in_grid.x;
                out[elem] = twice(values[elem]);
            """,
        )
        values = mx.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            self.call(helper, [values], values.shape, values.dtype).tolist(),
            [2.0, 4.0, 6.0, 8.0],
        )

        attribute = mx.fast.metal_kernel(
            name="omarchy_attribute",
            input_names=["values"],
            output_names=["result"],
            source="result[0] = threads_per_threadgroup.x;",
        )
        result = self.call(
            attribute,
            [values],
            (1,),
            mx.uint32,
            grid=(2, 1, 1),
            threadgroup=(2, 1, 1),
        )
        self.assertEqual(result.item(), 2)

    def test_same_name_different_source_in_one_batch(self):
        values = mx.arange(16, dtype=mx.float32)

        def apply(source):
            kernel = mx.fast.metal_kernel(
                name="omarchy_cache_key",
                input_names=["values"],
                output_names=["out"],
                source=source,
            )
            return self.call(kernel, [values], values.shape, values.dtype)

        doubled = apply(
            "uint elem = thread_position_in_grid.x; out[elem] = values[elem] * 2.0f;"
        )
        shifted = apply(
            "uint elem = thread_position_in_grid.x; out[elem] = values[elem] + 100.0f;"
        )
        mx.eval(doubled, shifted)
        self.assertEqual(doubled.tolist(), (values * 2).tolist())
        self.assertEqual(shifted.tolist(), (values + 100).tolist())

    def test_math_mode_and_mixed_dtypes(self):
        mode_source = """
            uint elem = thread_position_in_grid.x;
            #if defined(__FAST_MATH__) && __FAST_MATH__
            out[elem] = 1.0f;
            #else
            out[elem] = 0.0f;
            #endif
        """
        values = mx.zeros((4,), dtype=mx.float32)
        for mode, expected in (("safe", [0.0] * 4), ("fast", [1.0] * 4)):
            kernel = mx.fast.metal_kernel(
                name="omarchy_math_mode",
                input_names=["values"],
                output_names=["out"],
                source=mode_source,
                compile_options={"math_mode": mode},
            )
            self.assertEqual(
                self.call(kernel, [values], values.shape, values.dtype).tolist(),
                expected,
            )

        mixed = mx.fast.metal_kernel(
            name="omarchy_mixed_dtypes",
            input_names=["values"],
            output_names=["out"],
            source="""
                uint elem = thread_position_in_grid.x;
                out[elem] = values[elem] + values[elem];
            """,
        )
        half = mx.full((8,), 1.5, dtype=mx.float16)
        single = mx.full((8,), 2.5, dtype=mx.float32)
        total = self.call(mixed, [half], half.shape, half.dtype).astype(mx.float32)
        total = total + self.call(mixed, [single], single.shape, single.dtype)
        self.assertEqual(total.tolist(), [8.0] * 8)

    def test_threadgroup_memory_and_atomic_output(self):
        shared = mx.fast.metal_kernel(
            name="omarchy_shared",
            input_names=["values"],
            output_names=["out"],
            source="""
                threadgroup float scratch[4];
                uint lane = thread_position_in_threadgroup.x;
                scratch[lane] = values[lane];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) {
                    out[0] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
                }
            """,
        )
        values = mx.array([1.0, 2.0, 3.0, 4.0])
        result = self.call(
            shared,
            [values],
            (1,),
            mx.float32,
            grid=(4, 1, 1),
            threadgroup=(4, 1, 1),
        )
        self.assertEqual(result.item(), 10.0)

        atomic = mx.fast.metal_kernel(
            name="omarchy_atomic",
            input_names=["values"],
            output_names=["out"],
            atomic_outputs=True,
            source="""
                uint elem = thread_position_in_grid.x;
                atomic_fetch_add_explicit(
                    &out[0], values[elem], memory_order_relaxed);
            """,
        )
        result = self.call(
            atomic,
            [values],
            (1,),
            mx.float32,
            grid=(4, 1, 1),
            threadgroup=(4, 1, 1),
            init_value=0.0,
        )
        self.assertEqual(result.item(), 10.0)

    def test_unsupported_msl_is_named_refusal(self):
        kernel = mx.fast.metal_kernel(
            name="unsupported_texture",
            input_names=["values"],
            output_names=["out"],
            source="texture2d<float> image; out[0] = values[0];",
        )
        out = self.call(
            kernel,
            [mx.array([1.0])],
            (1,),
            mx.float32,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )
        with self.assertRaisesRegex(RuntimeError, "fast::CustomKernel MSL subset"):
            mx.eval(out)


if __name__ == "__main__":
    unittest.main()
