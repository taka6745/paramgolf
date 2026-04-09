# N-gram Logit Bias Submission

## Summary
Precomputed n-gram statistics (bigram, trigram, 4-gram) added as additive logit biases during training and evaluation. The neural model learns to predict RESIDUALS on top of classical n-gram knowledge instead of learning everything from scratch.

## Key Innovation
- Compute bigram P(next|prev) as direct 1024x1024 lookup table
- Compute trigram/4-gram P(next|context) using hash-based tables (8192 buckets)
- Add weighted log-probability biases to output logits: `logits += w * log P_ngram(next|context)`
- Weights: bigram=0.2, trigram=0.15, 4gram=0.1
- Model learns the residual: what n-grams can't predict

## Results (Mac MLX, 500 steps, 10 training shards)
- Baseline: 2.0239 bpb
- With n-gram bias (2-5gram, 65K buckets): **1.9428 bpb** (-0.081)
- With n-gram bias (2-5gram, 8K buckets, artifact-feasible): **1.9663 bpb** (-0.058)
- At 1000 steps: **1.8841 bpb** (-0.042 vs baseline@1000)

## Artifact Size
- Neural model (int8+zstd): ~5 MB
- Bigram table (int8+zstd): 0.4 MB
- Trigram table 8K (int8+zstd): 3.9 MB
- 4-gram table 8K (int8+zstd): 4.9 MB
- Total: ~14.2 MB (fits in 16 MB!)

## Architecture Stack
Built on top of existing SOTA techniques:
- LeakyReLU(0.5)^2
- SmearGate (bigram position blending)
- Weight decay 0.04 in Muon optimizer
- Depth recurrence (repeat encoder layers 3-4)
- All standard SOTA: 11L, 3xMLP, XSA, BigramHash, Partial RoPE, etc.

## Author
- taka6745 (Takoda Mundy)
