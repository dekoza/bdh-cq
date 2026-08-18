
import torch
from bdh_cq.bdh_cq import BDH, BDHReasoningWrapper

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

def test_bdh_reasoning_wrapper():

    model = BDH(
        dim = 512,
        num_tokens = 16,
        dim_qk_heads = 2048,
        depth = 2
    )

    wrapper = BDHReasoningWrapper(model)

    prompts = torch.randint(0, 16, (2, 20))
    answers = torch.randint(0, 16, (2, 30))

    answer_logits = wrapper(prompts, 8, answers)
    assert answer_logits.shape == (2, 30, 16)

    answer_logits = wrapper([prompts, 8, answers])
    assert answer_logits.shape == (2, 30, 16)

    # arbitrary stages: parallel -> latent -> parallel -> latent -> parallel

    p1 = torch.randint(0, 16, (1, 10))
    p2 = torch.randint(0, 16, (1, 15))
    ans = torch.randint(0, 16, (1, 20))

    ans_logits, memories = wrapper(p1, 2, p2, 4, ans, return_memory = True)

    assert ans_logits.shape == (1, 20, 16)
    assert memories.tokens_seen == (10 + 2 + 15 + 4 + 20)

def test_bdh_reasoning_wrapper_return_loss():

    model = BDH(
        dim = 512,
        num_tokens = 16,
        dim_qk_heads = 2048,
        depth = 2
    )

    wrapper = BDHReasoningWrapper(model)

    prompts = torch.randint(0, 16, (2, 20))
    answers = torch.randint(0, 16, (2, 30))

    loss, logits, memories = wrapper(prompts, 8, answers, return_loss = True, return_memory = True)

    assert logits.shape == (2, 30, 16)

    loss.backward()

def test_bdh_reasoning_wrapper_loss_without_latent():

    model = BDH(
        dim = 512,
        num_tokens = 16,
        dim_qk_heads = 2048,
        depth = 2
    )

    wrapper = BDHReasoningWrapper(model)

    prompts = torch.randint(0, 16, (2, 20))
    answers = torch.randint(0, 16, (2, 30))

    loss = wrapper(prompts, 0, answers, return_loss = True)

    loss.backward()

def test_bdh_reasoning_wrapper_loss_predicts_next_segment_first_token():

    # no answer targets: every latent token still predicts the first token
    # of the next tensor segment

    model = BDH(
        dim = 512,
        num_tokens = 16,
        dim_qk_heads = 2048,
        depth = 2
    )

    wrapper = BDHReasoningWrapper(model)

    p1 = torch.randint(0, 16, (2, 20))
    p2 = torch.randint(0, 16, (2, 30))

    loss = wrapper(p1, 8, p2, return_loss = True)

    loss.backward()

def test_bdh_reasoning_wrapper_loss_interleaved():

    # p1, 4 latent, p2, 5 latent, answer: each latent section predicts the
    # first token of the segment that follows it

    model = BDH(
        dim = 512,
        num_tokens = 16,
        dim_qk_heads = 2048,
        depth = 2
    )

    wrapper = BDHReasoningWrapper(model)

    p1 = torch.randint(0, 16, (2, 10))
    p2 = torch.randint(0, 16, (2, 15))
    answer = torch.randint(0, 16, (2, 20))

    loss, logits, memories = wrapper(p1, 4, p2, 5, answer, return_loss = True, return_memory = True)

    assert memories.tokens_seen == (10 + 4 + 15 + 5 + 20)
    assert logits.shape == (2, 20, 16)

    loss.backward()

def test_bdh_reasoning_wrapper_trailing_latent_rejected():

    model = BDH(
        dim = 512,
        num_tokens = 16,
        dim_qk_heads = 2048,
        depth = 2
    )

    wrapper = BDHReasoningWrapper(model)

    prompts = torch.randint(0, 16, (2, 20))

    try:
        wrapper(prompts, 8, return_loss = True)
        assert False, 'trailing latent reasoning should be rejected'
    except AssertionError:
        pass

def test_bdh_attn_residual_recycling():

    # pass the same sequence in again, attended over the previous pass's per-layer hiddens, alphafold2 style recycling

    model = BDH(
        dim = 512,
        num_tokens = 16,
        dim_qk_heads = 2048,
        depth = 2,
        attn_residual = True
    )

    tokens = torch.randint(0, 16, (1, 10))

    logits, _, per_pass_hiddens = model(tokens, return_memory = True, return_per_pass_hiddens = True)
    recycled = model(tokens, all_block_outputs = per_pass_hiddens)

    assert logits.shape == (1, 10, 16)
    assert recycled.shape == (1, 10, 16)

    # mismatched sequence length must be rejected

    try:
        model(tokens[:, :-1], all_block_outputs = per_pass_hiddens)
        assert False
    except AssertionError:
        pass
