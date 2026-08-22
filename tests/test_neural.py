import unittest

import numpy as np

try:
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    from misinformation_reliability.neural import (
        Vocabulary,
        build_word_model,
        class_weights,
        predict_bert_model,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "neural optional dependencies are not installed")
class NeuralTests(unittest.TestCase):
    def test_vocabulary_is_train_only_and_deterministic(self):
        first = Vocabulary.build(["Alpha beta beta", "gamma"], max_size=10, min_frequency=1)
        second = Vocabulary.build(["Alpha beta beta", "gamma"], max_size=10, min_frequency=1)
        self.assertEqual(first.token_to_id, second.token_to_id)
        encoded = first.encode("unseen beta", max_length=4)
        self.assertEqual(encoded[0], first.token_to_id["<unk>"])
        self.assertEqual(len(encoded), 4)

    def test_word_models_expose_binary_logits(self):
        vocabulary = Vocabulary.build(["one two three", "four five"], 20, 1)
        inputs = vocabulary.encode_many(["one two", "four five"], max_length=8)
        config = {
            "embedding_dim": 16,
            "fnn_hidden_dim": 8,
            "fnn_dropout": 0.1,
            "cnn_filters": 4,
            "cnn_kernel_sizes": [3, 4, 5],
            "cnn_dropout": 0.1,
            "lstm_hidden_dim": 8,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
        }
        for model_name in ("fnn", "cnn", "lstm"):
            logits = build_word_model(model_name, len(vocabulary), config)(inputs)
            self.assertEqual(tuple(logits.shape), (2, 2))

    def test_class_weights_balance_inverse_frequency(self):
        weights = class_weights(np.array([0, 1, 1, 1]))
        self.assertGreater(float(weights[0]), float(weights[1]))

    def test_bert_prediction_interface_without_network(self):
        config = BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=2,
        )
        model = BertForSequenceClassification(config)
        encodings = {
            "input_ids": torch.randint(1, 31, (4, 8)),
            "attention_mask": torch.ones((4, 8), dtype=torch.long),
        }
        predictions, probabilities, _ = predict_bert_model(
            model,
            encodings,
            np.array([0, 1, 0, 1]),
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))


if __name__ == "__main__":
    unittest.main()
