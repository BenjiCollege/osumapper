from __future__ import annotations

from typing import Any

from osumapper.errors import DependencyError, InputError


def _require_keras() -> Any:
    try:
        import keras
    except ImportError as exc:
        raise DependencyError(
            "Modern rhythm models require Keras. Run `uv sync --locked`."
        ) from exc
    return keras


def build_rhythm_model(
    *,
    audio_dimension: int,
    grid_dimension: int,
    sequence_length: int = 256,
    learning_rate: float = 1e-3,
    architecture: str = "transformer-v1",
    model_dimension: int | None = None,
    transformer_blocks: int | None = None,
    attention_heads: int | None = None,
    dropout: float = 0.15,
    positive_weight: float = 1.0,
    jit_compile: bool | str = "auto",
    weight_decay: float = 0.0,
    difficulty_dimension: int = 4,
) -> Any:
    if (
        audio_dimension <= 0
        or grid_dimension <= 0
        or sequence_length <= 0
        or difficulty_dimension <= 0
    ):
        raise InputError("Model dimensions must be positive.")
    keras = _require_keras()
    aliases = {
        "conv_audio_encoder_transformer": "transformer-v1",
        "transformer-v1": "transformer-v1",
        "conformer-v2": "conformer-v2",
        "conformer-v3": "conformer-v3",
        "conformer-v4": "conformer-v4",
        "conformer-v5": "conformer-v5",
    }
    try:
        selected_architecture = aliases[architecture.casefold()]
    except KeyError as exc:
        raise InputError(
            "Architecture must be transformer-v1 or conformer-v2 through conformer-v5."
        ) from exc
    conformer = selected_architecture.startswith("conformer-")
    dimension = model_dimension or (
        192
        if selected_architecture == "conformer-v5"
        else 160
        if selected_architecture in {"conformer-v3", "conformer-v4"}
        else 192
        if conformer
        else 128
    )
    blocks = transformer_blocks or (
        5 if selected_architecture == "conformer-v5" else 4 if conformer else 3
    )
    heads = attention_heads or (
        6
        if selected_architecture == "conformer-v5"
        else 5
        if selected_architecture in {"conformer-v3", "conformer-v4"}
        else 6
        if conformer
        else 4
    )
    if dimension % heads:
        raise InputError("Model dimension must be divisible by attention heads.")
    audio_input = keras.Input(
        (sequence_length, audio_dimension), name="audio_features", dtype="float32"
    )
    grid_input = keras.Input(
        (sequence_length, grid_dimension), name="grid_features", dtype="float32"
    )
    difficulty_input = keras.Input((difficulty_dimension,), name="difficulty", dtype="float32")
    if selected_architecture == "conformer-v5" and difficulty_dimension != 7:
        raise InputError("Conformer-v5 requires target stars plus six one-hot tier features.")

    audio = keras.layers.Conv1D(
        dimension, kernel_size=5, padding="same", activation="gelu", name="audio_encoder"
    )(audio_input)
    audio = keras.layers.LayerNormalization(name="audio_normalization")(audio)
    grid = keras.layers.Dense(dimension // 2, activation="gelu", name="grid_encoder")(grid_input)
    if selected_architecture == "conformer-v5":
        # Keep difficulty outside the shared song encoder. All six heads see
        # the same encoded music and grid; the one-hot tier input only selects
        # which head receives the loss for this curated human difficulty.
        combined = keras.layers.Concatenate(name="shared_feature_fusion")([audio, grid])
    else:
        difficulty = keras.layers.Dense(
            dimension // 2, activation="gelu", name="difficulty_encoder"
        )(difficulty_input)
        difficulty = keras.layers.RepeatVector(sequence_length, name="difficulty_broadcast")(
            difficulty
        )
        combined = keras.layers.Concatenate(name="feature_fusion")([audio, grid, difficulty])
    sequence = keras.layers.Dense(dimension, name="fusion_projection")(combined)
    attention_mask = None
    if selected_architecture in {"conformer-v3", "conformer-v4", "conformer-v5"}:
        # The musical grid is non-zero for every real candidate and zero for
        # padded positions, so this prevents padding from affecting attention.
        valid_steps = keras.ops.any(keras.ops.not_equal(grid_input, 0.0), axis=-1)
        attention_mask = keras.ops.logical_and(
            keras.ops.expand_dims(valid_steps, axis=1),
            keras.ops.expand_dims(valid_steps, axis=2),
        )

    if selected_architecture == "transformer-v1":
        for block in range(blocks):
            normalized = keras.layers.LayerNormalization(name=f"attention_norm_{block}")(sequence)
            attended = keras.layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=dimension // heads,
                dropout=dropout,
                name=f"self_attention_{block}",
            )(normalized, normalized)
            sequence = keras.layers.Add(name=f"attention_residual_{block}")([sequence, attended])
            normalized = keras.layers.LayerNormalization(name=f"feedforward_norm_{block}")(sequence)
            feedforward = keras.layers.Dense(
                dimension * 2, activation="gelu", name=f"feedforward_expand_{block}"
            )(normalized)
            feedforward = keras.layers.Dropout(dropout, name=f"feedforward_dropout_{block}")(
                feedforward
            )
            feedforward = keras.layers.Dense(dimension, name=f"feedforward_project_{block}")(
                feedforward
            )
            sequence = keras.layers.Add(name=f"feedforward_residual_{block}")(
                [sequence, feedforward]
            )
    else:
        for block in range(blocks):
            first_ff = keras.layers.LayerNormalization(name=f"conformer_ff1_norm_{block}")(sequence)
            first_ff = keras.layers.Dense(
                dimension * 4, activation="swish", name=f"conformer_ff1_expand_{block}"
            )(first_ff)
            first_ff = keras.layers.Dropout(dropout, name=f"conformer_ff1_dropout_{block}")(
                first_ff
            )
            first_ff = keras.layers.Dense(dimension, name=f"conformer_ff1_project_{block}")(
                first_ff
            )
            first_ff = keras.layers.Rescaling(0.5, name=f"conformer_ff1_scale_{block}")(first_ff)
            sequence = keras.layers.Add(name=f"conformer_ff1_residual_{block}")(
                [sequence, first_ff]
            )

            attention = keras.layers.LayerNormalization(name=f"conformer_attention_norm_{block}")(
                sequence
            )
            attention = keras.layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=dimension // heads,
                dropout=dropout,
                name=f"conformer_attention_{block}",
            )(attention, attention, attention_mask=attention_mask)
            attention = keras.layers.Dropout(dropout, name=f"conformer_attention_dropout_{block}")(
                attention
            )
            sequence = keras.layers.Add(name=f"conformer_attention_residual_{block}")(
                [sequence, attention]
            )

            convolution = keras.layers.LayerNormalization(
                name=f"conformer_convolution_norm_{block}"
            )(sequence)
            if selected_architecture in {"conformer-v3", "conformer-v4", "conformer-v5"}:
                convolution_value = keras.layers.Dense(
                    dimension,
                    name=f"conformer_convolution_value_{block}",
                )(convolution)
                convolution_gate = keras.layers.Dense(
                    dimension,
                    activation="sigmoid",
                    name=f"conformer_convolution_gate_{block}",
                )(convolution)
                convolution = keras.layers.Multiply(name=f"conformer_convolution_glu_{block}")(
                    [convolution_value, convolution_gate]
                )
                convolution_channels = dimension
            else:
                convolution = keras.layers.Dense(
                    dimension * 2,
                    activation="swish",
                    name=f"conformer_convolution_expand_{block}",
                )(convolution)
                convolution_channels = dimension * 2
            convolution = keras.layers.SeparableConv1D(
                convolution_channels,
                kernel_size=31,
                padding="same",
                name=f"conformer_depthwise_convolution_{block}",
            )(convolution)
            normalization = (
                keras.layers.LayerNormalization
                if selected_architecture in {"conformer-v3", "conformer-v4", "conformer-v5"}
                else keras.layers.BatchNormalization
            )
            convolution = normalization(name=f"conformer_convolution_batch_norm_{block}")(
                convolution
            )
            convolution = keras.layers.Activation(
                "swish", name=f"conformer_convolution_activation_{block}"
            )(convolution)
            convolution = keras.layers.Dense(
                dimension, name=f"conformer_convolution_project_{block}"
            )(convolution)
            convolution = keras.layers.Dropout(
                dropout, name=f"conformer_convolution_dropout_{block}"
            )(convolution)
            sequence = keras.layers.Add(name=f"conformer_convolution_residual_{block}")(
                [sequence, convolution]
            )

            second_ff = keras.layers.LayerNormalization(name=f"conformer_ff2_norm_{block}")(
                sequence
            )
            second_ff = keras.layers.Dense(
                dimension * 4, activation="swish", name=f"conformer_ff2_expand_{block}"
            )(second_ff)
            second_ff = keras.layers.Dropout(dropout, name=f"conformer_ff2_dropout_{block}")(
                second_ff
            )
            second_ff = keras.layers.Dense(dimension, name=f"conformer_ff2_project_{block}")(
                second_ff
            )
            second_ff = keras.layers.Rescaling(0.5, name=f"conformer_ff2_scale_{block}")(second_ff)
            sequence = keras.layers.Add(name=f"conformer_ff2_residual_{block}")(
                [sequence, second_ff]
            )

    sequence = keras.layers.LayerNormalization(name="output_normalization")(sequence)
    # Mixed precision keeps the expensive encoder on Tensor Cores, while a
    # float32 probability head avoids underflow in calibration and weighted BCE.
    if selected_architecture == "conformer-v5":
        tier_names = ("easy", "normal", "hard", "insane", "expert", "expert_plus")
        tier_outputs = []
        star_context = keras.layers.Dense(16, activation="gelu", name="target_star_encoder")(
            difficulty_input[:, :1]
        )
        star_context = keras.layers.RepeatVector(sequence_length, name="target_star_broadcast")(
            star_context
        )
        head_input = keras.layers.Concatenate(name="difficulty_head_input")(
            [sequence, star_context]
        )
        for index, tier_name in enumerate(tier_names):
            # Expert heads have extra capacity for complex rhythms while all
            # heads retain independent decision boundaries.
            head_dimension = dimension if index >= 4 else dimension // 2
            head = keras.layers.Dense(
                head_dimension,
                activation="gelu",
                name=f"{tier_name}_rhythm_head",
            )(head_input)
            head = keras.layers.Dropout(dropout, name=f"{tier_name}_head_dropout")(head)
            tier_outputs.append(
                keras.layers.Dense(
                    1,
                    activation="sigmoid",
                    dtype="float32",
                    name=f"{tier_name}_probability",
                )(head)
            )
        all_tiers = keras.layers.Concatenate(name="difficulty_head_probabilities")(tier_outputs)
        tier_selector = keras.layers.RepeatVector(sequence_length, name="tier_selector_broadcast")(
            difficulty_input[:, 1:]
        )
        selected_tier = keras.layers.Multiply(name="select_difficulty_head")(
            [all_tiers, tier_selector]
        )
        selected_tier = keras.ops.sum(selected_tier, axis=-1, keepdims=True)
        output = keras.layers.Activation("linear", dtype="float32", name="hit_probability")(
            selected_tier
        )
    else:
        output = keras.layers.Dense(
            1, activation="sigmoid", dtype="float32", name="hit_probability"
        )(sequence)
    model = keras.Model(
        inputs={
            "audio_features": audio_input,
            "grid_features": grid_input,
            "difficulty": difficulty_input,
        },
        outputs=output,
        name=(
            f"osumapper_rhythm_{selected_architecture.replace('-', '_')}"
            if conformer
            else "osumapper_modern_rhythm_v1"
        ),
    )
    compile_rhythm_model(
        model,
        learning_rate=learning_rate,
        positive_weight=positive_weight,
        jit_compile=jit_compile,
        optimizer_name=(
            "adamw"
            if selected_architecture in {"conformer-v3", "conformer-v4", "conformer-v5"}
            else "adam"
        ),
        weight_decay=weight_decay,
    )
    return model


def _sequence_f1_metric(keras: Any) -> Any:
    @keras.saving.register_keras_serializable(package="osumapper")
    class SequenceBinaryF1(keras.metrics.Metric):
        def __init__(self, name: str = "f1", threshold: float = 0.5, **kwargs: Any) -> None:
            super().__init__(name=name, **kwargs)
            self.threshold = threshold
            self.true_positives = self.add_weight(name="tp", initializer="zeros")
            self.false_positives = self.add_weight(name="fp", initializer="zeros")
            self.false_negatives = self.add_weight(name="fn", initializer="zeros")

        def update_state(self, y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> None:
            ops = keras.ops
            actual = ops.cast(y_true > 0.5, self.dtype)
            predicted = ops.cast(y_pred >= self.threshold, self.dtype)
            if sample_weight is None:
                weight = ops.ones_like(actual)
            else:
                weight = ops.cast(sample_weight, self.dtype)
                if len(weight.shape) == len(actual.shape) - 1:
                    weight = ops.expand_dims(weight, axis=-1)
                weight = ops.broadcast_to(weight, ops.shape(actual))
            self.true_positives.assign_add(ops.sum(weight * actual * predicted))
            self.false_positives.assign_add(ops.sum(weight * (1.0 - actual) * predicted))
            self.false_negatives.assign_add(ops.sum(weight * actual * (1.0 - predicted)))

        def result(self) -> Any:
            denominator = 2.0 * self.true_positives + self.false_positives + self.false_negatives
            return keras.ops.divide_no_nan(2.0 * self.true_positives, denominator)

        def reset_state(self) -> None:
            self.true_positives.assign(0.0)
            self.false_positives.assign(0.0)
            self.false_negatives.assign(0.0)

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "threshold": self.threshold}

    return SequenceBinaryF1()


def _weighted_binary_crossentropy(keras: Any, positive_weight: float) -> Any:
    @keras.saving.register_keras_serializable(package="osumapper")
    class WeightedBinaryCrossentropy(keras.losses.Loss):
        def __init__(
            self,
            positive_weight: float = 1.0,
            name: str = "weighted_binary_crossentropy",
            **kwargs: Any,
        ) -> None:
            super().__init__(name=name, **kwargs)
            self.positive_weight = float(positive_weight)

        def call(self, y_true: Any, y_pred: Any) -> Any:
            ops = keras.ops
            actual = ops.cast(y_true, y_pred.dtype)
            probability = ops.clip(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
            return -(
                self.positive_weight * actual * ops.log(probability)
                + (1.0 - actual) * ops.log(1.0 - probability)
            )

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "positive_weight": self.positive_weight}

    return WeightedBinaryCrossentropy(positive_weight=positive_weight)


def compile_rhythm_model(
    model: Any,
    *,
    learning_rate: float,
    positive_weight: float = 1.0,
    jit_compile: bool | str = "auto",
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> None:
    keras = _require_keras()
    if optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=1.0,
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=_weighted_binary_crossentropy(keras, positive_weight),
        metrics=[
            keras.metrics.Precision(name="precision", thresholds=0.5),
            keras.metrics.Recall(name="recall", thresholds=0.5),
            _sequence_f1_metric(keras),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
        jit_compile=jit_compile,
    )


def load_rhythm_model(path: Any, *, compile_model: bool = False) -> Any:
    keras = _require_keras()
    try:
        return keras.models.load_model(path, compile=compile_model)
    except Exception as exc:
        raise InputError(f"Could not load modern rhythm model {path}: {exc}") from exc
