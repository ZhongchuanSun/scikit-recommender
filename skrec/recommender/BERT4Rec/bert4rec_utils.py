import pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf
from . import modeling, optimization
from ...utils.py import MetricReport, EarlyStopping
from ...utils.py.cython import eval_score_matrix


class EvalHooks(tf.train.SessionRunHook):
    def __init__(self, config, user_history_filename, evaluator, logger):
        if user_history_filename is not None:
            # print('load user history from :' + user_history_filename)
            with open(user_history_filename, 'rb') as input_file:
                user_history = pickle.load(input_file)
        self.config = config
        self.max_pre_seq = int(round(self.config.max_seq_len*self.config.masked_lm_prob))

        self.user_train_dict = {}
        self.user_test_dict = {}
        for user, items in user_history.items():
            user = int(user.split("_")[1])
            # items = items[0]
            self.user_train_dict[user] = np.array(items[:-1], dtype=np.int32)
            self.user_test_dict[user] = set(items[-1:])

        self.seq_len_num = defaultdict(int)
        for user, item_seq in self.user_train_dict.items():
            self.seq_len_num[len(item_seq)] += 1

        self._evaluator = evaluator
        self._logger = logger

        self._logger.info("metrics:".ljust(12)+f"\t{self._evaluator.metrics_str}")
        # self._logger.info(self._evaluator.metrics_info())
        self._epoch = 0
        self.early_stopping = EarlyStopping(metric="NDCG@10", patience=self.config.early_stop)

    def begin(self):
        self._eval_results = []

    def end(self, session):
        all_user_result = np.concatenate(self._eval_results, axis=0)  # (num_users, metrics_num*max_top)
        final_result = np.mean(all_user_result, axis=0)  # (1, metrics_num*max_top)
        final_result = np.reshape(final_result, newshape=[self._evaluator.metrics_num, self._evaluator.max_top])  # (metrics_num, max_top)
        final_result = final_result[:, self._evaluator.top_show - 1]
        final_result = np.reshape(final_result, newshape=[-1])
        cur_result = MetricReport(self._evaluator.metrics_list, final_result)
        self._logger.info(f"test:".ljust(12)+f"\t{cur_result.values_str}")

        self.counter += 1
        if self._epoch >= 80 and self.early_stopping(cur_result):
            self._logger.info("early stop")
            self._logger.info("best:".ljust(12)+f"\t{self.early_stopping.best_result.values_str}")
            exit(0)

        self._epoch += 1

    def before_run(self, run_context):
        variables = tf.get_collection('eval_sp')
        return tf.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        masked_lm_log_probs, info = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, self.max_pre_seq, masked_lm_log_probs.shape[1]))

        bat_ratings = np.array(masked_lm_log_probs[:, 0, :], dtype=np.float32, copy=True)
        batch_users = np.squeeze(info)

        test_items = [self.user_test_dict[u] for u in batch_users]

        for idx, user in enumerate(batch_users):
            if user in self.user_train_dict and len(self.user_train_dict[user]) > 0:
                train_items = self.user_train_dict[user]
                bat_ratings[idx, train_items] = -np.inf

        result = eval_score_matrix(bat_ratings, test_items, self._evaluator.metrics,
                                   top_k=self._evaluator.max_top, thread_num=self._evaluator.num_thread)  # (B,k*metric_num)

        self._eval_results.append(result)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config,
            model.get_sequence_output(),
            model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            tf.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.add_to_collection('eval_sp', info)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=None,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % mode)

        return output_spec

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info":
                tf.FixedLenFeature([1], tf.int64),  #[user]
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)

        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example
