"""
dpo_toy_demo.py
──────────────────
A small, from-scratch implementation of Direct Preference Optimization
(DPO) on a tiny word-level LSTM language model. See README.md in this
directory for what this does and doesn't demonstrate.

Pipeline
----------
1. Generate synthetic (prompt, chosen, rejected) preference triples:
   prompt = a bare headline fact, chosen = a concise/neutral summary of
   it, rejected = a verbose/sensational summary of the same fact.
2. Build a tiny word-level vocabulary and tokenize everything to fixed
   prompt/response lengths.
3. Build a policy LM (Embedding -> LSTM -> Dense(vocab)) and take a
   frozen deep copy of its *initial* weights as the reference model.
4. Train the policy with the real DPO loss:
     L = -log(sigmoid(beta * ((logp_pi(chosen|x) - logp_ref(chosen|x))
                             - (logp_pi(rejected|x) - logp_ref(rejected|x)))))
5. Report the DPO loss curve and the preference margin
   (logp(chosen) - logp(rejected)) under the policy vs. under the frozen
   reference, before and after training.
"""

import random

import numpy as np
import tensorflow as tf

SEED = 7
PROMPT_LEN = 6
RESPONSE_LEN = 10
EMBED_DIM = 32
LSTM_UNITS = 64
BETA = 0.5
LEARNING_RATE = 5e-3
EPOCHS = 60
BATCH_SIZE = 16

PAD_ID = 0
UNK_ID = 1

TICKERS = ["ACME", "NORTH", "BLUE", "SUMMIT", "DELTA", "ORION"]
EVENTS = [
    ("beat earnings estimates", "beats estimates"),
    ("missed revenue targets", "misses targets"),
    ("announced a new product line", "launches new product"),
    ("cut its full year guidance", "cuts guidance"),
    ("raised its dividend", "raises dividend"),
    ("completed a share buyback", "completes buyback"),
]

VERBOSE_TEMPLATES = [
    "BREAKING: {ticker} stuns markets as it {fact}, sending shockwaves through Wall Street",
    "In a dramatic turn of events {ticker} {fact} and investors are scrambling to react",
    "{ticker} sent the market into a frenzy after it {fact} this morning",
]
CONCISE_TEMPLATES = [
    "{ticker} {short}",
    "{ticker}: {short}",
    "{ticker} today {short}",
]
PROMPT_TEMPLATES = [
    "{ticker} {fact}",
]


def _tokenize(text: str):
    return text.lower().replace(":", "").split()


def generate_preference_data(seed: int = SEED):
    rng = random.Random(seed)
    triples = []
    for ticker in TICKERS:
        for fact, short in EVENTS:
            prompt = rng.choice(PROMPT_TEMPLATES).format(ticker=ticker, fact=fact)
            chosen = rng.choice(CONCISE_TEMPLATES).format(ticker=ticker, short=short)
            rejected = rng.choice(VERBOSE_TEMPLATES).format(ticker=ticker, fact=fact)
            triples.append((prompt, chosen, rejected))
    rng.shuffle(triples)
    return triples


def build_vocab(triples):
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for prompt, chosen, rejected in triples:
        for text in (prompt, chosen, rejected):
            for tok in _tokenize(text):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, length: int):
    ids = [vocab.get(tok, UNK_ID) for tok in _tokenize(text)]
    ids = ids[:length]
    ids = ids + [PAD_ID] * (length - len(ids))
    return ids


def build_model(vocab_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(PROMPT_LEN + RESPONSE_LEN - 1,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, EMBED_DIM, mask_zero=False)(inputs)
    x = tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True)(x)
    logits = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs, logits)


def sequence_logprob(model, full_ids: tf.Tensor, response_mask: tf.Tensor) -> tf.Tensor:
    """
    full_ids: (batch, PROMPT_LEN + RESPONSE_LEN) int32 token ids.
    response_mask: (batch, PROMPT_LEN + RESPONSE_LEN - 1) 1.0 where the
      *target* position (t+1) is a real, non-pad response token.
    Returns: (batch,) sum of log p(target_t | tokens_<=t) over masked positions
      — i.e. logp(response | prompt) under teacher forcing.
    """
    inputs = full_ids[:, :-1]
    targets = full_ids[:, 1:]
    logits = model(inputs, training=True)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    target_log_probs = tf.gather(log_probs, targets, batch_dims=2)
    masked = target_log_probs * response_mask
    return tf.reduce_sum(masked, axis=1)


def dpo_loss(policy, reference, batch, beta: float):
    full_ids, mask_chosen, mask_rejected, full_ids_rejected = batch

    logp_pi_chosen = sequence_logprob(policy, full_ids, mask_chosen)
    logp_pi_rejected = sequence_logprob(policy, full_ids_rejected, mask_rejected)
    logp_ref_chosen = sequence_logprob(reference, full_ids, mask_chosen)
    logp_ref_rejected = sequence_logprob(reference, full_ids_rejected, mask_rejected)

    pi_logratios = logp_pi_chosen - logp_pi_rejected
    ref_logratios = logp_ref_chosen - logp_ref_rejected
    logits = beta * (pi_logratios - ref_logratios)

    loss = -tf.math.log_sigmoid(logits)
    margin = pi_logratios  # policy's current preference margin, for logging
    return tf.reduce_mean(loss), tf.reduce_mean(margin)


def make_batches(triples, vocab):
    full_chosen, full_rejected, mask_chosen, mask_rejected = [], [], [], []
    total_len = PROMPT_LEN + RESPONSE_LEN

    for prompt, chosen, rejected in triples:
        prompt_ids = encode(prompt, vocab, PROMPT_LEN)
        chosen_ids = encode(chosen, vocab, RESPONSE_LEN)
        rejected_ids = encode(rejected, vocab, RESPONSE_LEN)

        full_chosen.append(prompt_ids + chosen_ids)
        full_rejected.append(prompt_ids + rejected_ids)

        # mask over target positions (length total_len - 1): position t
        # predicts token t+1; we score t+1 in [PROMPT_LEN, total_len-1]
        # and require it not be padding.
        m_chosen = np.zeros(total_len - 1, dtype="float32")
        m_rejected = np.zeros(total_len - 1, dtype="float32")
        for t in range(PROMPT_LEN - 1, total_len - 1):
            if full_chosen[-1][t + 1] != PAD_ID:
                m_chosen[t] = 1.0
            if full_rejected[-1][t + 1] != PAD_ID:
                m_rejected[t] = 1.0
        mask_chosen.append(m_chosen)
        mask_rejected.append(m_rejected)

    return (
        np.array(full_chosen, dtype="int32"),
        np.array(mask_chosen, dtype="float32"),
        np.array(mask_rejected, dtype="float32"),
        np.array(full_rejected, dtype="int32"),
    )


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    triples = generate_preference_data()
    vocab = build_vocab(triples)
    print(f"{len(triples)} preference pairs, vocab size {len(vocab)}")

    full_chosen, mask_chosen, mask_rejected, full_rejected = make_batches(triples, vocab)

    policy = build_model(len(vocab))
    policy.build(input_shape=(None, PROMPT_LEN + RESPONSE_LEN - 1))

    reference = build_model(len(vocab))
    reference.build(input_shape=(None, PROMPT_LEN + RESPONSE_LEN - 1))
    reference.set_weights(policy.get_weights())  # frozen copy of the INITIAL policy
    reference.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    n = len(triples)

    def eval_margin(model):
        _, margin = dpo_loss(
            model, reference,
            (full_chosen, mask_chosen, mask_rejected, full_rejected),
            BETA,
        )
        return float(margin)

    print(f"Policy preference margin before training: {eval_margin(policy):.4f}")

    indices = np.arange(n)
    for epoch in range(EPOCHS):
        np.random.shuffle(indices)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            batch = (
                full_chosen[batch_idx],
                mask_chosen[batch_idx],
                mask_rejected[batch_idx],
                full_rejected[batch_idx],
            )
            with tf.GradientTape() as tape:
                loss, _ = dpo_loss(policy, reference, batch, BETA)
            grads = tape.gradient(loss, policy.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy.trainable_variables))
            epoch_losses.append(float(loss))

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  dpo_loss={np.mean(epoch_losses):.4f}")

    print(f"Policy preference margin after training:  {eval_margin(policy):.4f}")
    print(
        "\nA rising margin (logp(chosen) - logp(rejected) under the policy) means "
        "the model increasingly assigns higher likelihood to the concise/neutral "
        "response over the verbose/sensational one, relative to where the frozen "
        "reference started."
    )


if __name__ == "__main__":
    main()
