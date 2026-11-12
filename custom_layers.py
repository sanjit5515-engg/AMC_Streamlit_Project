import tensorflow as tf
from tensorflow.keras.layers import (
    LayerNormalization, Dense, Conv1D, Dropout
)
from tensorflow.keras.initializers import LecunNormal


class SelectiveSSM(tf.keras.layers.Layer):
    """Custom Selective State Space Model layer used during AMC training."""

    def __init__(self, d_model=128, d_state=16, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state

    def build(self, input_shape):
        self.x_proj = Dense(
            self.d_model + 2 * self.d_state,
            kernel_initializer=LecunNormal()
        )
        self.dt_proj = Dense(
            self.d_model,
            kernel_initializer=LecunNormal()
        )
        self.log_A = self.add_weight(
            name="log_A",
            shape=(self.d_model, self.d_state),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.02
            ),
            trainable=True
        )
        self.D = self.add_weight(
            name="D",
            shape=(self.d_model,),
            initializer="ones",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]

        x_proj = self.x_proj(inputs)
        delta_raw = x_proj[:, :, :self.d_model]
        B = x_proj[:, :, self.d_model:self.d_model + self.d_state]
        C = x_proj[:, :, self.d_model + self.d_state:]

        delta = tf.nn.softplus(self.dt_proj(delta_raw))
        A = -tf.exp(self.log_A)

        delta_expanded = delta[:, :, :, None]
        A_expanded = A[None, None, :, :]
        delta_A = tf.exp(delta_expanded * A_expanded)

        B_expanded = B[:, :, None, :]
        x_expanded = inputs[:, :, :, None]
        delta_B_x = delta_expanded * B_expanded * x_expanded

        delta_A = tf.transpose(delta_A, [1, 0, 2, 3])
        delta_B_x = tf.transpose(delta_B_x, [1, 0, 2, 3])
        C_time = tf.transpose(C, [1, 0, 2])

        initial_state = tf.zeros(
            (batch_size, self.d_model, self.d_state),
            dtype=inputs.dtype
        )
        outputs = tf.TensorArray(dtype=inputs.dtype, size=sequence_length)

        def condition(t, state, output_array):
            return t < sequence_length

        def body(t, state, output_array):
            current_A = delta_A[t]
            current_Bx = delta_B_x[t]
            current_C = C_time[t]

            state = current_A * state + current_Bx
            y = tf.reduce_sum(
                state * current_C[:, None, :],
                axis=-1
            )
            output_array = output_array.write(t, y)
            return t + 1, state, output_array

        _, _, outputs = tf.while_loop(
            condition,
            body,
            (tf.constant(0), initial_state, outputs),
            maximum_iterations=sequence_length
        )

        y = tf.transpose(outputs.stack(), [1, 0, 2])
        return y + inputs * self.D


class MambaBlock(tf.keras.layers.Layer):
    """Mamba block matching the architecture used to train the models."""

    def __init__(
        self,
        d_model=128,
        d_state=16,
        d_conv=4,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.norm = LayerNormalization(epsilon=1e-5)
        self.input_projection = Dense(
            2 * d_model,
            kernel_initializer=LecunNormal()
        )
        self.conv = Conv1D(
            filters=d_model,
            kernel_size=d_conv,
            padding="same",
            groups=d_model,
            activation="swish"
        )
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state
        )
        self.output_projection = Dense(
            d_model,
            kernel_initializer=LecunNormal()
        )
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        residual = inputs
        x = self.norm(inputs)
        projected = self.input_projection(x)
        x, z = tf.split(projected, 2, axis=-1)
        x = self.conv(x)
        x = tf.nn.silu(x)
        x = self.ssm(x)
        z = tf.nn.silu(z)
        x = x * z
        x = self.output_projection(x)
        x = self.dropout(x, training=training)
        return residual + x
