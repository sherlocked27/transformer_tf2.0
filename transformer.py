import numpy as np
from icecream import ic
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model) -> None:
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, inputs, **kwargs):
        max_length = inputs.shape[1]
        emb = self.embedding(inputs)
        # [[1],[2]] + np.expand_dims([1], axis=0) = array([[2],[3]])
        emb += self.positionalEncoding(max_length, self.d_model)
        return emb

    def positionalEncoding(max_len, d_emb):
        pe = np.array([
            [pos / (10000 ** ((2 * i)/d_emb)) for i in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
        ])
        pe[:, ::2] = np.sin(pe[:, ::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, axis=0)  # To incoperate the batch size
        return pe


class ScaledSingleAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledSingleAttention, self).__init__()
        self.d_h = d_h

    def call(self, q, k, v, mask=None, **kwargs):
        qdotk = tf.matmul(q, k, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention = qdotk/scale
        if(mask is not None):
            scaled_attention = scaled_attention + (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention)

        return tf.matmul(attention_weights, v), attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_h, n_h, dropout, **kwargs):
        super().__init__()
        self.d_h = d_h
        self.n_h = n_h
        self.dropout = dropout
        self.ssa = ScaledSingleAttention(d_h)
        self.dense = tf.keras.layers.Dense(self.d_h)

    def call(self, query, key, value, mask=None, **kwargs):
        batch_size = tf.shape(query)[0]
        dim_att = self.d_h // self.n_h
        w_key = tf.keras.layers.Dense(self.d_h, use_bias=False)
        w_value = tf.keras.layers.Dense(self.d_h, use_bias=False)
        w_query = tf.keras.layers.Dense(self.d_h, use_bias=False)

        key = w_key(key)
        query = w_query(query)
        value = w_value(value)

        key_h = self.reshape_for_multihead(key, batch_size, dim_att)
        query_h = self.reshape_for_multihead(query, batch_size, dim_att)
        value_h = self.reshape_for_multihead(value, batch_size, dim_att)

        atten_output, attention_weights = self.ssa(
            query_h, key_h, value_h, mask=mask)
        ic(atten_output.shape)
        concatenated = tf.reshape(tf.transpose(atten_output, [0, 2, 1, 3]), [
                                  batch_size, -1, self.d_h])
        output = self.dense(concatenated)

        return output, attention_weights

    def reshape_for_multihead(self, vector, batch_size, dim_att):
        new_vector = tf.reshape(vector, [batch_size, -1, self.n_h, dim_att])
        ic(new_vector.shape)
        return tf.transpose(new_vector, [0, 2, 1, 3])


class PositionFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.d1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.d2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs, **kwargs):
        input = self.d1(inputs)
        ic(input.shape)
        output = self.d2(input)
        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_h, d_ff, dropout, ** kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, n_h, dropout)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.position_feed_forward = PositionFeedForward(d_model, d_ff)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=0.0001)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=0.0001)

    def call(self, inputs, padding_mask, training=False, **kwargs):
        # Self Attention
        attended_output, attention_weights = self.mha(
            inputs, inputs, inputs, mask=padding_mask)
        ic(attended_output.shape)
        attended_output_drop = self.dropout_1(
            attended_output, training=training)
        attended_output_norm = self.layer_norm_1(attended_output_drop+inputs)

        # Feed Forward
        output_encoder_unnorm = self.position_feed_forward(
            attended_output_norm)
        output_encoder_unnorm_drop = self.dropout_2(
            output_encoder_unnorm, training=training)
        output_encoder = self.layer_norm_2(
            output_encoder_unnorm_drop+attended_output_norm)

        return output_encoder


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_h, d_ff, dropout, ** kwargs):
        super().__init__(**kwargs)

        self.mha_mask = MultiHeadAttention(d_model, n_h, dropout)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=0.0001)

        self.mha = MultiHeadAttention(d_model, n_h, dropout)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=0.0001)

        self.position_feed_forward = PositionFeedForward(d_model, d_ff)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=0.0001)

        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
        self.dropout_3 = tf.keras.layers.Dropout(dropout)

    def call(self, encoder_output, inputs, padding_mask, look_ahead_mask=None, training=False, ** kwargs):
        # Masked self attention
        attended_output_mask, attention_weights_mask = self.mha_mask(
            inputs, inputs, inputs, mask=look_ahead_mask)
        attended_output_drop_mask = self.dropout_1(
            attended_output_mask, training=training)
        attended_output_norm_mask = self.layer_norm_1(
            attended_output_drop_mask+inputs)

        # Self attention
        attended_output, attention_weights = self.mha(
            attended_output_norm_mask, encoder_output, encoder_output, mask=padding_mask)
        attended_output_drop = self.dropout_2(
            attended_output, training=training)
        attended_output_norm = self.layer_norm_2(
            attended_output_drop + attended_output_norm_mask)

        # feed forward
        output_decoder_unnorm = self.position_feed_forward(
            attended_output_norm)
        output_decoder_unnorm_drop = self.dropout_3(
            output_decoder_unnorm, training=training)
        output_decoder = self.layer_norm_3(
            output_decoder_unnorm_drop + attended_output_norm)

        return output_decoder


class Mask:
    def look_ahead_mask(cls, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    def padding_mask(cls, sequences):
        """
        This is important as  Keeping the embedding of <pad> as a constant zero vector is sorta important.
        Therefore every sentence has diffrent length and we take care of that
        """
        sequences = tf.cast(tf.math.equal(sequences, 0), tf.float64)
        return sequences[:, tf.newaxis, tf.newaxis, :]

    def create_all_mask(cls, seq_len, encoder_sequences, decoder_sequence):
        """
        For look ahead mask tf.maximum is important as the actual length might be 
        smaller than the sequence length and we dont want padding to influence our attention criteria.
        For the very same reason we do masking for encoder and decoder 2nd attention layer too 
        """
        encoder_padding_mask = cls.padding_mask(encoder_sequences)
        decoder_padding_mask = cls.padding_mask(decoder_sequence)

        look_ahead_mask = tf.maximum(cls.look_ahead_mask(seq_len=seq_len),
                                     cls.padding_mask(sequences=decoder_sequence))

        return encoder_padding_mask, look_ahead_mask, decoder_padding_mask


class Transformer(tf.keras.layers.Layer):
    def __init__(self,  vocab_size, d_model, n_h, d_ff, dropout, encoder_count, decoder_count, ** kwargs):
        super().__init__(**kwargs)
        self.encoder_embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.decoder_embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.encoders = []
        self.decoders = []
        for _ in range(encoder_count):
            self.encoders.append(Encoder(d_model, n_h, d_ff, dropout))
        for _ in range(encoder_count):
            self.decoders.append(Decoder(d_model, n_h, d_ff, dropout))

        self.linear = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, encoder_mask, decoder_mask, look_ahead_mask, decoder_input, training=False, **kwargs):
        inputs = self.encoder_embedding_layer(inputs)
        for i in self.encoder_count:
            encoder_output = self.encoders[i](
                inputs, mask=encoder_mask, training=training)
            inputs = encoder_output

        decoder_input = self.decoder_embedding_layer(decoder_input)
        for i in self.decoder_count:
            decoder_output = self.decoders[i](
                encoder_output, decoder_input, decoder_mask, look_ahead_mask, training=training)
            decoder_input = decoder_output

        return self.linear(decoder_output)
