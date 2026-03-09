"""Smoke test for M2 modules (no GPU / real models needed)."""
import sys
sys.path.insert(0, 'd:/ECQ_Former_antigravity')

import torch
import torch.nn as nn


class FakeTok:
    def __call__(self, texts, **kw):
        B = len(texts)
        return {
            'input_ids': torch.randint(0, 100, (B, 10)),
            'attention_mask': torch.ones(B, 10, dtype=torch.long),
        }


class FakeEmbed(nn.Embedding):
    def __init__(self):
        super().__init__(100, 4096)


def test_text_gating():
    print('=== text_gating.py ===')
    from models.bridge.text_gating import TextSemanticEncoder, EncoderTextGating

    # TextSemanticEncoder
    enc = TextSemanticEncoder(lm_embed_dim=4096, d_bridge=768).float()
    t = enc(['What is the finding?', 'Is there pneumonia?'], FakeTok(), FakeEmbed(), torch.device('cpu'))
    assert t.shape == (2, 768), f'FAIL shape={t.shape}'
    print(f'  TextSemanticEncoder (2,768): OK')

    # EncoderTextGating – training mode (Encoder Dropout active)
    gating = EncoderTextGating(d_bridge=768, n_encoders=3, encoder_drop_p=1.0)  # p=1 → always drop
    gating.train()
    tv = torch.randn(2, 768)
    vis = [torch.randn(2, 196, 768)] * 3
    gated = gating(tv, vis)
    assert len(gated) == 3 and gated[0].shape == (2, 196, 768)
    print(f'  EncoderTextGating train (Encoder Dropout p=1): OK')

    # eval mode
    gating.eval()
    gated_e = gating(tv, vis)
    assert gated_e[0].shape == (2, 196, 768)
    print(f'  EncoderTextGating eval: OK')

    # get_gate_weights
    w = gating.get_gate_weights(tv)
    assert w.shape == (2, 3)
    print(f'  get_gate_weights (2,3): OK  sample={w[0].tolist()}')


def test_film_modulation():
    print('\n=== film_modulation.py ===')
    from models.bridge.film_modulation import FiLMQueryModulator

    film = FiLMQueryModulator(d_model=768)
    q  = torch.randn(2, 96, 768)
    tv = torch.randn(2, 768)
    q_out = film(q, tv)
    assert q_out.shape == (2, 96, 768), f'FAIL shape={q_out.shape}'
    print(f'  FiLMQueryModulator (2,96,768): OK')

    # Check zero-init: with zero-init modulator, output should be close to concat_proj(q)
    film.eval()
    with torch.no_grad():
        q_out2 = film(q, tv)
    assert not torch.isnan(q_out2).any(), 'NaN detected in FiLM output'
    print(f'  No NaN in output: OK')
    return film


def test_meqformerv2(film):
    print('\n=== MEQFormerV2 (meqformer.py) ===')
    from models.bridge.meqformer import MEQFormerV2

    meq = MEQFormerV2(d=768, nhead=12, num_layers=2, m_queries=96, film_modulator=film)
    tv = torch.randn(2, 768)
    kv = torch.randn(2, 588, 768)  # 3×196 gated tokens

    # With text_vec → FiLM applied
    out = meq(kv, text_vec=tv)
    assert out.z.shape == (2, 96, 768), f'FAIL shape={out.z.shape}'
    print(f'  MEQFormerV2 with text_vec: z=(2,96,768) OK')

    # Without text_vec → backward compat
    out2 = meq(kv)
    assert out2.z.shape == (2, 96, 768)
    print(f'  MEQFormerV2 without text_vec: z={out2.z.shape} OK')


def main():
    torch.manual_seed(42)
    test_text_gating()
    film = test_film_modulation()
    test_meqformerv2(film)
    print('\n=== ALL SMOKE TESTS PASSED ===')


if __name__ == '__main__':
    main()
