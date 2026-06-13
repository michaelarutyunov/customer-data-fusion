# Clickstream Encoder SPEC

## Purpose

Encodes web session event sequences (browsing behaviour) into a 128-dim customer embedding. Captures the *how customers search online* signal — session depth, browsing patterns, research intensity, and purchase funnel behaviour.

## Architecture

```
Per session:  [T_events x 19]  → token_proj(19→64) → GRU(64→128, 2 layers) → session hidden [128]
Customer:     mean-pool sessions → output_proj(128→128) + LayerNorm → e_clickstream [128]
```

## Tokenisation (19-dim per event)

| Feature | Derivation | Dim |
|---------|-----------|-----|
| `event_type_embed` | Learned embedding, vocab = 8 ClickstreamEventType values | 8 |
| `page_type_embed` | Learned embedding, vocab = 6 PageType values | 6 |
| `device_embed` | Learned embedding, vocab = 3 DeviceType values | 3 |
| `dwell_log` | `log1p(dwell_ms / 1000.0)` | 1 |
| `is_purchase` | 1.0 if `event_type == PURCHASE` else 0.0 | 1 |

## Session → Customer aggregation

- Sessions encoded individually via GRU (final hidden state of last layer)
- Session embeddings mean-pooled per customer (with mask for padding)
- Truncation: most recent 50 sessions, 40 events per session

## Training

- **Objective**: CE (7-class archetype) + NT-Xent (individual identity, split-view contrastive)
- **Batch size**: 256
- **LR**: 1e-3
- **Epochs**: 50
- **Split**: by customer, 80/20 train/val

## Output Contract

```python
e_clickstream = encoder(session_embeddings, session_mask)  # [B, EMBEDDING_DIM=128]
```

## Dependencies

- Requires clickstream generator output (`data/synthetic/clickstream/session_events.jsonl`)
- Anonymous sessions (customer_id='anonymous') excluded from training
