import os
import collections
import random
import numpy as np

import tensorflow as tf

from .vocab import FreqVocab
import pickle
import multiprocessing
import time
from ...io import RSDataset

random_seed = 12345
short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([printable_text(x) for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        try:
            input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        except:
            print(instance)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["info"] = create_int_feature(instance.info)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)
    return total_written


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(all_documents_raw,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              rng,
                              vocab,
                              mask_prob,
                              sliding_step,
                              pool_size,
                              force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}

    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue
            all_documents[user] = [item_seq[-max_num_tokens:]]
    else:
        max_num_tokens = max_seq_length  # we need two sentence

        assert sliding_step > 0
        assert sliding_step < max_num_tokens
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue

            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                beg_idx = list(range(len(item_seq)-max_num_tokens, 0, -sliding_step))
                beg_idx.append(0)
                all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

    instances = []
    if force_last:
        for user in all_documents:
            instances.extend(
                create_instances_from_document_test(
                    all_documents, user, max_seq_length))
        print("num of instance:{}".format(len(instances)))
    else:
        start_time = time.perf_counter()
        pool = multiprocessing.Pool(processes=pool_size)
        instances = []
        print("document num: {}".format(len(all_documents)))

        def log_result(result):
            print("callback function result type: {}, size: {} ".format(type(result), len(result)))
            instances.extend(result)

        for step in range(dupe_factor):
            # create_instances_threading(
            #     all_documents, user, max_seq_length, short_seq_prob,
            #     masked_lm_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1, 10000)),
            #     mask_prob, step)
            pool.apply_async(
                create_instances_threading, args=(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1,10000)),
                    mask_prob, step), callback=log_result)
        pool.close()
        pool.join()

        for user in all_documents:
            instances.extend(mask_last(all_documents, user, max_seq_length, short_seq_prob,
                                       masked_lm_prob, max_predictions_per_seq, vocab, rng))

        print("num of instance:{}; time:{}".format(len(instances), time.perf_counter() - start_time))
    rng.shuffle(instances)
    return instances


def create_instances_threading(all_documents, user, max_seq_length, short_seq_prob,
                               masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
    cnt = 0
    start_time = time.perf_counter()
    instances = []
    vocab_item_set = set(vocab.get_items())
    for user in all_documents:
        cnt += 1
        if cnt % 5000 == 0:
            print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt, time.perf_counter()-start_time))
            start_time = time.perf_counter()
        instances.extend(create_instances_from_document_train(
            all_documents, user, max_seq_length, short_seq_prob,
            masked_lm_prob, max_predictions_per_seq, vocab_item_set, rng,
            mask_prob))

    return instances


def mask_last(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1
        assert len(tokens) <= max_num_tokens

        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    assert len(document) == 1 and len(document[0]) <= max_num_tokens

    tokens = document[0]
    assert len(tokens) >= 1

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

    info = [int(user.split("_")[1])]
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    return [instance]


def create_instances_from_document_train(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab_items, rng, mask_prob):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]

    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    # vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1
        assert len(tokens) <= max_num_tokens

        (tokens, masked_lm_positions, masked_lm_labels) = \
            create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_items, rng, mask_prob)

        instance = TrainingInstance(info=info,
                                    tokens=tokens,
                                    masked_lm_positions=masked_lm_positions,
                                    masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)
    output_tokens[last_index] = "[MASK]"

    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token not in vocab_words:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_prob:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            raise NotImplementedError
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                raise NotImplementedError
                masked_token = rng.choice(vocab_words)


        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def gen_samples(data,
                output_filename,
                rng,
                vocab,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                mask_prob,
                masked_lm_prob,
                max_predictions_per_seq,
                sliding_step,
                pool_size,
                force_last=False):
    # create train
    instances = create_training_instances(
        data, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, rng, vocab, mask_prob, sliding_step,
        pool_size, force_last)

    tf.logging.info("*** Writing to output files ***")
    tf.logging.info("  %s", output_filename)

    total_written = write_instance_to_example_files(instances, max_seq_length,
                                                    max_predictions_per_seq, vocab,
                                                    [output_filename])
    return total_written


def main(config, dataset: RSDataset, output_dir: str, tf_record_name: str):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    max_seq_length = config.max_seq_len
    max_predictions_per_seq = int(round(config.max_seq_len*config.masked_lm_prob))
    masked_lm_prob = config.masked_lm_prob
    mask_prob = 1.0
    dupe_factor = config.dupe_factor
    sliding_step = config.sliding_step
    pool_size = config.pool_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_train = dataset.train_data.to_user_dict_by_time()
    user_train = {user: items.tolist() for user, items in user_train.items()}

    user_test = dataset.test_data.to_user_dict_by_time()
    user_test = {user: items.tolist() for user, items in user_test.items()}

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }
    rng = random.Random(random_seed)

    vocab = FreqVocab(user_test_data)
    user_test_data_output = {
        k: vocab.convert_tokens_to_ids(v)
        for k, v in user_test_data.items()
    }

    print('begin to generate train')
    output_filename = os.path.join(output_dir, tf_record_name + '.train.tfrecord')
    total_written = gen_samples(user_train_data,
                                output_filename,
                                rng,
                                vocab,
                                max_seq_length,
                                dupe_factor,
                                short_seq_prob,
                                mask_prob,
                                masked_lm_prob,
                                max_predictions_per_seq,
                                sliding_step,
                                pool_size,
                                force_last=False)
    print('train:{}'.format(output_filename))
    np.save(os.path.join(output_dir, tf_record_name + '.train.num.npy'), total_written)

    print('begin to generate test')
    output_filename = os.path.join(output_dir, tf_record_name + '.test.tfrecord')
    gen_samples(
        user_test_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        -1.0,
        pool_size,
        force_last=True)
    print('test:{}'.format(output_filename))

    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
          format(vocab.get_vocab_size(),
                 vocab.get_user_count(),
                 vocab.get_item_count(),
                 vocab.get_item_count() + vocab.get_special_token_count()))
    vocab_file_name = os.path.join(output_dir, tf_record_name + '.vocab')
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = os.path.join(output_dir, tf_record_name + '.his')
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    print('done.')
