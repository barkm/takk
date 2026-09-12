from isolated_sign_validation.parallel import parallel_map


def test_parallel_map_keeps_order_across_batches():
    items = list(range(-50, 50))
    assert list(parallel_map(abs, items, batch_size=7, chunksize=3)) == [abs(i) for i in items]
