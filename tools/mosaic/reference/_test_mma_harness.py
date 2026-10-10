
    return mgpu.FragmentedArray.splat(
        arith_dialect.index_cast(ir.IntegerType.get_signless(32), grid_idx),
        shape=shape,
        layout=layout.to_mgpu(),
        is_signed=False
    )
  return fn()


def _array_splat(value, shape: tuple[int, ...]):
  """Same as `jnp.full(shape, value, jnp.float32)` but implemented using `inline_mgpu`.

  This is useful to prevent the result from being optimized away.
  """
  @plgpu.inline_mgpu(
      return_type=plgpu.ShapeDtypeStruct(
          shape, jnp.float32, layout=plgpu.Layout.WG_SPLAT(shape)
      ),
  )
  def fn(_):
    ir_value = mgpu.c(value, ir.F32Type.get())
    return mgpu.FragmentedArray.splat(ir_value, shape)
  return fn()


class PallasTestMetaclass(parameterized.TestGeneratorMetaclass):

  def __new__(mcs, *args, lowering_semantics=plgpu.LoweringSemantics.Lane):
    cls = super().__new__(mcs, *args)
    cls.LOWERING_SEMANTICS = lowering_semantics
    return cls


class MonkeyPatchTest(jtu.JaxTestCase):

  def test_calling_kernel_directly_raises(self):
    with self.assertRaises(RuntimeError):
      plgpu.kernel()


def run_on_sm80(method):
  """A marker that allows a test case to run on Ampere GPUs."""
  method._min_capability = "8.0"
  return method


class PallasTest(jtu.JaxTestCase, metaclass=PallasTestMetaclass):
  LOWERING_SEMANTICS: ClassVar[plgpu.LoweringSemantics]

  def setUp(
      self, *, artificial_shared_memory_limit=jtu._SMEM_SIZE_BOUND_FOR_TESTS
  ):
    if jtu.test_device_matches(["rocm"]):
      self.skipTest("Mosaic GPU is not supported on ROCm.")
    min_capability = getattr(
        getattr(type(self), self._testMethodName, None),
        "_min_capability",
        "9.0",
    )
    if not jtu.is_cuda_compute_capability_at_least(min_capability):
      self.skipTest(f"Only works on a GPU with capability >= sm{min_capability}")

    super().setUp()
    self.enter_context(mgpu.core.artificial_shared_memory_limit(artificial_shared_memory_limit))

  def is_wg_semantics(self):
    return self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup

  def skip_if_wg_semantics(self):
    if self.is_wg_semantics():
      self.skipTest("Not supported under WG semantics")

  def kernel(self, *args, **kwargs):
    compiler_params = dataclasses.replace(
        kwargs.pop("compiler_params", plgpu.CompilerParams()),
        lowering_semantics=self.LOWERING_SEMANTICS,
    )
    return _kernel(*args, compiler_params=compiler_params, **kwargs)

  def pallas_call(
      self,
      fn,
      *,
      out_shape,
      grid=(),
      in_specs=(),
      out_specs=(),
      scratch_shapes=(),
      compiler_params=plgpu.CompilerParams(),
  ):
    compiler_params = dataclasses.replace(
        compiler_params,
        lowering_semantics=self.LOWERING_SEMANTICS,
    )
    from jax._src.pallas.mosaic_gpu import pallas_call
    return pallas_call.pallas_call(
        fn,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=compiler_params,
        kernel_fn=self.kernel,
    )

  @contextlib.contextmanager
  def capture_stdout(self):
    if "pytest" in sys.modules:
      self.skipTest("pytest interacts badly with GPU stdout capture")
    if mosaic_gpu_lib is None:
      raise ValueError("Running tests but missing Mosaic GPU extension")
    with jtu.capture_stdout() as stdout:
      yield stdout
      # We need to cudaDeviceSynchronize to make sure printfs are flushed.
      mosaic_gpu_lib._mosaic_gpu_ext._sync_all_devices()

  def default_transforms(
      self, *, swizzle: int = 128, dtype: jnp.dtype
  ) -> Sequence[plgpu.Transform]:
