# Campaign Encoder SPEC

## Purpose

Encodes campaign interaction sequences (email/push response behaviour) into a 128-dim customer embedding. Captures the *intervention response* signal — how customers react to marketing campaigns, which is the modality that closes the counterfactual loop for offer escalation scenarios.

## Architecture

```
Input: [T_campaigns x 11]  → input_proj(11→32) → positional encoding
      → Transformer encoder (2 heads, 2 layers, d_model=32, d_ff=128)
      → CLS token [32] → cls_proj(32→128) → e_campaign [128]
```

## Tokenisation (11-dim per event)

| Feature | Derivation | Dim |
|---------|-----------|-----|
| `campaign_type_embed` | Learned embedding, vocab = 5 CampaignType values | 5 |
| `discount_pct` | Direct, 0.0–0.5 | 1 |
| `funnel_flags` | opened, clicked, converted, unsub (each 0.0/1.0) | 4 |

## Sequence handling

- All campaign events within training window aggregated into one chronological sequence
- Truncation: most recent 50 campaigns
- CLS token prepended at index 0 (learned embedding via positional encoding)

## Training

- **Objective**: CE (7-class archetype) + NT-Xent (individual identity)
- **Batch size**: 256
- **LR**: 1e-3
- **Epochs**: 50
- **Split**: by customer, 80/20 train/val
- **Strategy recovery target**: >50% (lower than other encoders — campaigns carry less discriminative signal)

## Output Contract

```python
e_campaign = encoder(tokens, mask)  # [B, EMBEDDING_DIM=128]
```

## Dependencies

- Requires campaign generator output (`data/synthetic/campaigns.jsonl`)
- Customers who unsubscribed early have shorter sequences (handled by padding + mask)
