import numpy as np

EPS = 1e-8


def make_moons_np(n_samples=1000, noise=0.24, seed=7):
    """Return X shape (N, 2), y shape (N, 1)."""
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    t0 = rng.random(n0) * np.pi
    t1 = rng.random(n1) * np.pi

    x0 = np.column_stack([np.cos(t0), np.sin(t0)])
    x1 = np.column_stack([1.0 - np.cos(t1), -np.sin(t1) + 0.5])

    X = np.vstack([x0, x1])
    X += noise * rng.standard_normal(X.shape)

    y = np.concatenate([np.zeros(n0), np.ones(n1)])[:, None]

    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


def train_val_split(X, y, val_fraction=0.5):
    n = X.shape[0]
    n_val = int(n * val_fraction)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    return X_train, y_train, X_val, y_val


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def d_relu(x):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    a = np.tanh(x)
    return 1.0 - a * a


def binary_cross_entropy(y_true, y_prob):
    y_prob = np.clip(y_prob, EPS, 1.0 - EPS)
    return -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))


def l2_penalty(params):
    total = 0.0
    L = len(params) // 2
    for l in range(1, L + 1):
        total += np.sum(params[f"W{l}"] ** 2)
    return total


def accuracy(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return np.mean(y_pred == y_true)


ACTIVATIONS = {
    "relu": (relu, d_relu),
    "tanh": (tanh, d_tanh),
}


def init_mlp(layer_sizes, seed=0, hidden_activation="relu"):
    """layer_sizes example: [2, 32, 32, 1]."""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(layer_sizes)):
        fan_in = layer_sizes[l - 1]
        if l == len(layer_sizes) - 1:
            scale = np.sqrt(1.0 / fan_in)
        elif hidden_activation == "relu":
            scale = np.sqrt(2.0 / fan_in)
        elif hidden_activation == "tanh":
            scale = np.sqrt(1.0 / fan_in)
        else:
            raise ValueError(f"Unsupported hidden_activation: {hidden_activation}")

        params[f"W{l}"] = rng.standard_normal((fan_in, layer_sizes[l])) * scale
        params[f"b{l}"] = np.zeros((1, layer_sizes[l]))
    return params


def forward_pass(X, params, hidden_activation="relu"):
    """
    Returns:
      y_prob: output probabilities, shape (N, 1)
      cache: intermediates for backprop
    """
    act_fn, _ = ACTIVATIONS[hidden_activation]
    cache = {"A0": X}
    L = len(params) // 2

    A = X
    for l in range(1, L):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        A = act_fn(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

    ZL = A @ params[f"W{L}"] + params[f"b{L}"]
    AL = sigmoid(ZL)
    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = AL
    return AL, cache


def backward_pass(y_true, params, cache, hidden_activation="relu", l2_lambda=0.0):
    grads = {}
    _, d_act_fn = ACTIVATIONS[hidden_activation]
    L = len(params) // 2
    N = y_true.shape[0]

    A_prev = cache[f"A{L-1}"]
    AL = cache[f"A{L}"]
    dZ = AL - y_true
    grads[f"dW{L}"] = (A_prev.T @ dZ) / N + (l2_lambda / N) * params[f"W{L}"]
    grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True) / N

    for l in range(L - 1, 0, -1):
        dA = dZ @ params[f"W{l+1}"].T
        dZ = dA * d_act_fn(cache[f"Z{l}"])
        A_prev = cache[f"A{l-1}"]
        grads[f"dW{l}"] = (A_prev.T @ dZ) / N + (l2_lambda / N) * params[f"W{l}"]
        grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / N

    return grads


def update_params(params, grads, lr=0.05):
    L = len(params) // 2
    for l in range(1, L + 1):
        params[f"W{l}"] -= lr * grads[f"dW{l}"]
        params[f"b{l}"] -= lr * grads[f"db{l}"]


def iterate_minibatches(X, y, batch_size=64, shuffle=True, seed=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    layer_sizes=(2, 32, 32, 1),
    epochs=400,
    lr=0.05,
    batch_size=64,
    seed=123,
    hidden_activation="relu",
    l2_lambda=0.0,
    early_stopping=False,
    patience=25,
    min_delta=0.0,
    restore_best=True,
    verbose=True,
):
    params = init_mlp(list(layer_sizes), seed=seed, hidden_activation=hidden_activation)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs_ran": 0,
    }

    best_val_loss = np.inf
    best_params = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        batch_seed = seed + epoch
        for xb, yb in iterate_minibatches(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=batch_seed,
        ):
            yb_prob, cache = forward_pass(xb, params, hidden_activation=hidden_activation)
            grads = backward_pass(
                yb,
                params,
                cache,
                hidden_activation=hidden_activation,
                l2_lambda=l2_lambda,
            )
            update_params(params, grads, lr=lr)

        train_prob, _ = forward_pass(X_train, params, hidden_activation=hidden_activation)
        val_prob, _ = forward_pass(X_val, params, hidden_activation=hidden_activation)

        train_loss = binary_cross_entropy(y_train, train_prob) + 0.5 * l2_lambda * l2_penalty(params) / X_train.shape[0]
        val_loss = binary_cross_entropy(y_val, val_prob) + 0.5 * l2_lambda * l2_penalty(params) / X_val.shape[0]
        train_acc = accuracy(y_train, train_prob)
        val_acc = accuracy(y_val, val_prob)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epochs_ran"] = epoch

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in params.items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print(
                f"epoch {epoch:4d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

        if early_stopping and patience_counter >= patience:
            if verbose:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, best val_loss={best_val_loss:.4f})"
                )
            break

    if early_stopping and restore_best and best_params is not None:
        params = best_params

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    return params, history


def prepare_data(n_samples=1200, noise=0.24, seed=7):
    X, y = make_moons_np(n_samples=n_samples, noise=noise, seed=seed)
    return train_val_split(X, y, val_fraction=0.2)


def summarize_run(label, history):
    idx = int(np.argmin(history["val_loss"]))
    print(label)
    print(f"  epochs_ran:      {history['epochs_ran']}")
    print(f"  best_epoch:      {idx + 1}")
    print(f"  final_train_loss {history['train_loss'][-1]:.4f}")
    print(f"  final_val_loss   {history['val_loss'][-1]:.4f}")
    print(f"  final_train_acc  {history['train_acc'][-1]:.3f}")
    print(f"  final_val_acc    {history['val_acc'][-1]:.3f}")
