
  # TODO(slebedev): Support unsigned int types.
  @parameterized.product(
      dtype=(
          jnp.bfloat16,
          jnp.float16,
          jnp.float8_e4m3fn,
          jnp.float8_e5m2,
          jnp.int8,
          jnp.int4,
      ),
  )
  def test_mma(self, dtype):
    m = k = 128
    n = 8
    dtype = jnp.dtype(dtype)
    is_integer = jnp.issubdtype(dtype, jnp.integer)
    acc_dtype = jnp.int32 if is_integer else jnp.float32

    @functools.partial(
        self.kernel,
        out_type=jax.ShapeDtypeStruct((m, n), acc_dtype),
    )
    def kernel(a_ref, b_ref, o_ref):
      acc = plgpu.layout_cast(
          jnp.zeros((m, n), acc_dtype), plgpu.Layout.MMA_ACC(dtype)
      )
      a = plgpu.load(a_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False)
      b = plgpu.load(
          b_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False
      )
      o_ref[...] = plgpu.mma(acc, a, b)

    prng = np.random.default_rng(0)
    if is_integer:
      iinfo = jnp.iinfo(dtype)
      a = prng.integers(iinfo.min, iinfo.max + 1, (m, k)).astype(dtype)
      b = prng.integers(iinfo.min, iinfo.max + 1, (n, k)).astype(dtype)
    else:
      a = prng.uniform(-1, 1, (m, k)).astype(dtype)
      b = prng.uniform(-1, 1, (n, k)).astype(dtype)
