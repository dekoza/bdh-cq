
def test_bdh_cq():
    import torch
    from bdh_cq.bdh_cq import BDH

    model = BDH(
        dim = 512,
        num_tokens = 256
    )

    ids = torch.randint(0, 256, (2, 1024))

    logits = model(ids)

    assert logits.shape == (2, 1024, 256)
