
import torch
from bdh_cq.bdh_cq import BDH

def test_bdh_cq():

    model = BDH(
        dim = 512,
        num_tokens = 256
    )

    ids = torch.randint(0, 256, (2, 1024))

    logits = model(ids)

    assert logits.shape == (2, 1024, 256)

    logits, memories = model(ids, return_memory = True)
    logits2 = model(ids, memories = memories)

    assert logits2.shape == logits.shape

def test_bdh_cq_latent_reasoning():

    model = BDH(
        dim = 512,
        num_tokens = 16
    )

    prompts = torch.randint(0, 16, (1, 50))
    answers = torch.randint(0, 16, (1, 100))

    _, memories = model(prompts, return_memory = True)

    latent = memories.embeds[..., -1:, :]
    for _ in range(8):
        _, memories = model(latent, memories = memories, return_memory = True, return_logits = False, update_memory = False)
        latent = memories.embeds

    answer_logits = model(answers, memories = memories)

    assert answer_logits.shape == (1, 100, 16)
