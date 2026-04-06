"""
Multi-Order N-gram Eval Cache + Hedge Mixer
============================================
Drop-in module for competition eval pipeline.
Builds n-gram cache from scored tokens, combines with model via hedge mixer.

Usage:
    cache = NgramEvalCache(max_order=7, min_count=2)
    mixer = HedgeMixer(n_experts=2, eta=0.1)
    
    for pos in range(len(tokens)):
        model_probs = model.predict(tokens[:pos])
        cache_probs = cache.predict(tokens[:pos])
        
        # Combine
        combined = mixer.mix([model_probs, cache_probs])
        
        # Score
        actual = tokens[pos]
        loss = -log(combined[actual])
        
        # Update (score-first: update AFTER scoring)
        mixer.update([model_probs[actual], cache_probs[actual]])
        cache.update(tokens[:pos+1])
"""

import numpy as np
from collections import defaultdict


class NgramEvalCache:
    """Multi-order n-gram cache built during eval."""
    
    def __init__(self, max_order=7, min_count=2, vocab_size=8192):
        self.max_order = max_order
        self.min_count = min_count
        self.V = vocab_size
        # cache[order][context_tuple] = {next_token: count}
        self.cache = {k: defaultdict(lambda: defaultdict(int)) 
                      for k in range(2, max_order + 1)}
    
    def predict(self, context_tokens):
        """Return probability distribution from cache, or None if no match."""
        probs = np.zeros(self.V) + 1.0 / self.V  # uniform prior
        
        for order in range(min(self.max_order, len(context_tokens)), 1, -1):
            ctx = tuple(context_tokens[-order+1:]) if order > 1 else ()
            if ctx in self.cache[order]:
                counts = self.cache[order][ctx]
                total = sum(counts.values())
                if total >= self.min_count:
                    # Smooth with Laplace
                    for tok, cnt in counts.items():
                        probs[tok] = (cnt + 0.1) / (total + 0.1 * self.V)
                    # Re-normalize
                    probs = probs / probs.sum()
                    return probs
        
        return probs  # uniform if no match
    
    def update(self, tokens):
        """Add n-grams from the token sequence to the cache."""
        n = len(tokens)
        if n < 2:
            return
        # Only add n-grams ending at the last token (incremental)
        for order in range(2, min(self.max_order + 1, n + 1)):
            ctx = tuple(tokens[-(order):-(1)])
            self.cache[order][ctx][tokens[-1]] += 1


class HedgeMixer:
    """Adaptive hedge mixer for combining multiple predictors."""
    
    def __init__(self, n_experts=2, eta=0.1):
        self.n_experts = n_experts
        self.eta = eta
        self.weights = np.ones(n_experts) / n_experts
    
    def mix(self, expert_probs):
        """Combine expert probabilities using current weights."""
        combined = np.zeros_like(expert_probs[0])
        for w, p in zip(self.weights, expert_probs):
            combined += w * p
        return combined / combined.sum()
    
    def update(self, expert_correct_probs):
        """Update weights based on expert performance on the actual token."""
        losses = np.array([-np.log(p + 1e-15) for p in expert_correct_probs])
        self.weights *= np.exp(-self.eta * losses)
        self.weights /= self.weights.sum()


# Quick self-test
if __name__ == "__main__":
    print("Testing NgramEvalCache + HedgeMixer...")
    
    cache = NgramEvalCache(max_order=7, min_count=1, vocab_size=256)
    mixer = HedgeMixer(n_experts=2, eta=0.1)
    
    # Simulate scoring some text
    text = b"the cat sat on the mat the cat sat on the mat again"
    tokens = list(text)
    
    correct = 0
    total = 0
    
    for i in range(1, len(tokens)):
        # Model prediction (uniform for this test)
        model_p = np.ones(256) / 256
        # Cache prediction
        cache_p = cache.predict(tokens[:i])
        # Combined
        combined = mixer.mix([model_p, cache_p])
        
        actual = tokens[i]
        total += 1
        if np.argmax(combined) == actual:
            correct += 1
        
        # Update
        mixer.update([model_p[actual], cache_p[actual]])
        cache.update(tokens[:i+1])
    
    print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"  Final weights: model={mixer.weights[0]:.3f}, cache={mixer.weights[1]:.3f}")
    print(f"  Cache entries: {sum(len(c) for c in cache.cache.values())}")
    print("  PASSED!")
