__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["PointwiseIterator", "PairwiseIterator", "InteractionIterator",
           "SequentialPointwiseIterator", "SequentialPairwiseIterator",
           "UserVecIterator", "ItemVecIterator",
           "KGPairwiseIterator"
           ]

from typing import Dict
from collections import Iterable
from collections import OrderedDict
import numpy as np
from ..utils.py import OrderedDefaultDict
from ..utils.py import BatchIterator
from ..utils.py import randint_choice
from ..utils.py import pad_sequences
from .dataset import ImplicitFeedback, KnowledgeGraph
from .dataset import _HEAD, _RELATION, _TAIL


class _Iterator(object):
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def _generate_positive_items(user_pos_dict: Dict[int, np.ndarray]):
    assert user_pos_dict, "'user_pos_dict' cannot be empty."

    list_users, list_items = [], []
    user_n_pos = OrderedDict()

    for user, items in user_pos_dict.items():
        list_items.append(items)
        list_users.append(np.full_like(items, user))
        user_n_pos[user] = len(items)
    users_ary = np.concatenate(list_users, axis=0)
    items_ary = np.concatenate(list_items, axis=0)
    return user_n_pos, users_ary, items_ary


def _generative_time_order_positive_items(user_pos_dict: Dict[int, np.ndarray],
                                          num_previous: int=1, num_next: int=1, pad=None):
    assert user_pos_dict, "'user_pos_dict' cannot be empty."
    assert num_previous >= 1
    assert num_next >= 1
    # TODO 需要充分的测试.
    users_list, seqs_list = [], []
    user_n_pos = OrderedDefaultDict(int)

    tot_len = num_previous + num_next
    for user, seq_items in user_pos_dict.items():
        # user_n_pos[user] = 0
        for idx in range(len(seq_items), 0, -1):
            cur_seqs = seq_items[:idx]
            if len(cur_seqs) >= tot_len:  # not pad
                seqs_list.append(cur_seqs[-tot_len:])
                users_list.append(user)
                user_n_pos[user] += 1
            elif pad is not None and len(cur_seqs) > num_next:  # pad
                seqs_list.append(cur_seqs[-tot_len:])
                users_list.append(user)
                user_n_pos[user] += 1
            else:  # next user
                break

    if pad is not None and tot_len > 2:
        seqs_ary = pad_sequences(seqs_list, value=pad, max_len=tot_len,
                                 padding='pre', truncating='pre', dtype=np.int32)
    else:
        seqs_ary = np.int32(seqs_list)

    previous_items, next_items = np.split(seqs_ary, [num_previous], axis=-1)
    users_ary = np.int32(users_list)
    return user_n_pos, users_ary, previous_items, next_items


def _sampling_negative_items(user_n_pos: OrderedDict, num_neg: int, num_items: int,
                             user_pos_dict: Dict[int, np.ndarray]):
    assert num_neg > 0, "'num_neg' must be a positive integer."

    neg_items_list = []
    for user, n_pos in user_n_pos.items():
        neg_items = randint_choice(num_items, size=n_pos*num_neg, exclusion=user_pos_dict[user])
        if num_neg == 1:
            neg_items = neg_items if isinstance(neg_items, Iterable) else np.int32([neg_items])
        else:
            neg_items = np.reshape(neg_items, newshape=[n_pos, num_neg])
        neg_items_list.append(neg_items)

    return np.concatenate(neg_items_list)


class InteractionIterator(_Iterator):
    def __init__(self, dataset: ImplicitFeedback, batch_size: int = 1024,
                 shuffle: bool = True, drop_last: bool = False):
        super(InteractionIterator, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        ui_pairs = dataset.to_user_item_pairs()
        self.users = ui_pairs[:, 0]
        self.pos_items = ui_pairs[:, 1]

    def __len__(self):
        n_sample = len(self.users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        data_iter = BatchIterator(self.users, self.pos_items,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_items)


class PointwiseIterator(_Iterator):
    """Sample negative items and iterate dataset with pointwise training instances.

    The training instances consist of `batch_users`, `batch_items` and
    `batch_labels`, which are lists of users, items and labels. All lengths of
    them are `batch_size`.
    Positive and negative items are labeled as `1.` and  `0.`, respectively.
    """
    def __init__(self, dataset: ImplicitFeedback,
                 num_neg: int=1, batch_size: int=1024,
                 shuffle: bool=True, drop_last: bool=False):
        """Initializes a new `PointwiseIterator` instance.

        Args:
            dataset (ImplicitFeedback): An instance of `Interaction`.
            num_neg (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(PointwiseIterator, self).__init__()
        assert num_neg > 0, "'num_neg' must be a positive integer."

        self.num_neg = num_neg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()

        self.user_n_pos, users_ary, self.pos_items = \
            _generate_positive_items(self.user_pos_dict)

        self.all_users = np.tile(users_ary, self.num_neg + 1)
        num_pos_items = len(self.pos_items)
        pos_labels = np.ones(num_pos_items, dtype=np.float32)
        neg_labels = np.zeros(num_pos_items*self.num_neg, dtype=np.float32)
        self.all_labels = np.concatenate([pos_labels, neg_labels], axis=0)

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        neg_items = _sampling_negative_items(self.user_n_pos, self.num_neg,
                                             self.num_items, self.user_pos_dict)
        # neg_items.shape: [-1, num_neg]

        neg_items = neg_items.transpose().reshape([-1])
        all_items = np.concatenate([self.pos_items, neg_items], axis=0)

        data_iter = BatchIterator(self.all_users, all_items, self.all_labels,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_items, bat_labels in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_items), np.asarray(bat_labels)


class PairwiseIterator(_Iterator):
    def __init__(self, dataset: ImplicitFeedback, num_neg: int=1,
                 batch_size: int=1024, shuffle: bool=True, drop_last: bool=False):
        """Initializes a new `PairwiseSampler` instance.

        Args:
            dataset (data.Interaction): An instance of `data.Interaction`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(PairwiseIterator, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.drop_last = drop_last
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()

        self.user_n_pos, self.all_users, self.pos_items = \
            _generate_positive_items(self.user_pos_dict)

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        neg_items = _sampling_negative_items(self.user_n_pos, self.num_neg,
                                             self.num_items, self.user_pos_dict)

        data_iter = BatchIterator(self.all_users, self.pos_items, neg_items,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items)


class SequentialPointwiseIterator(_Iterator):
    def __init__(self, dataset: ImplicitFeedback,
                 num_previous: int=1, num_next: int=1, num_neg: int=1,
                 pad: int=None, batch_size: int=1024,
                 shuffle: bool=True, drop_last: bool=False):
        super(SequentialPointwiseIterator, self).__init__()
        assert num_previous >= 1
        assert num_next >= 1
        assert num_neg >= 1
        # TODO 确保这个实现是正确的, 和最新的序列化算法中的实现对比.
        self.num_previous = num_previous
        self.num_next = num_next
        self.num_neg = num_neg
        self.pad = pad
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict_by_time()

        self.user_n_pos, users_ary, item_seqs_ary, self.pos_next_items = \
            _generative_time_order_positive_items(self.user_pos_dict, num_previous=num_previous,
                                                  num_next=num_next, pad=pad)

        self.all_users = np.tile(users_ary, self.num_neg + 1)
        self.all_item_seqs = np.tile(item_seqs_ary, [self.num_neg + 1, 1]).squeeze()

        len_pos = len(self.pos_next_items)
        pos_labels = np.ones([len_pos, num_next], dtype=np.float32)
        neg_labels = np.zeros([len_pos*self.num_neg, num_next], dtype=np.float32)
        self.all_labels = np.concatenate([pos_labels, neg_labels], axis=0).squeeze()

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        neg_next_items = _sampling_negative_items(self.user_n_pos, self.num_neg * self.num_next,
                                                  self.num_items, self.user_pos_dict)

        neg_next_items = np.concatenate(np.split(neg_next_items, self.num_neg, axis=-1), axis=0)
        all_next_items = np.concatenate([self.pos_next_items, neg_next_items], axis=0).squeeze()

        data_iter = BatchIterator(self.all_users, self.all_item_seqs,
                                  all_next_items, self.all_labels,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_item_seqs, bat_next_items, bat_labels in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_item_seqs), \
                  np.asarray(bat_next_items), np.asarray(bat_labels)


class SequentialPairwiseIterator(_Iterator):
    def __init__(self, dataset: ImplicitFeedback,
                 num_previous: int = 1, num_next: int = 1,
                 pad: int = None, batch_size: int = 1024,
                 shuffle: bool = True, drop_last: bool = False):
        assert num_previous >= 1
        assert num_next >= 1
        # TODO 确保这个实现是正确的, 和最新的序列化算法中的实现对比.
        self.num_previous = num_previous
        self.num_next = num_next
        self.pad = pad
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict_by_time()

        self.user_n_pos, self.all_users, all_item_seqs, pos_next_items = \
            _generative_time_order_positive_items(self.user_pos_dict, num_previous=num_previous,
                                                  num_next=num_next, pad=pad)

        self.all_item_seqs = all_item_seqs.squeeze()
        self.pos_next_items = pos_next_items.squeeze()

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        neg_next_items = _sampling_negative_items(self.user_n_pos, self.num_next, self.num_items,
                                                  self.user_pos_dict).squeeze()

        data_iter = BatchIterator(self.all_users, self.all_item_seqs,
                                  self.pos_next_items, neg_next_items,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_item_seqs, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_item_seqs), \
                  np.asarray(bat_pos_items), np.asarray(bat_neg_items)


class UserVecIterator(_Iterator):
    def __init__(self, dataset: ImplicitFeedback, batch_size: int=1024,
                 shuffle: bool=True, drop_last: bool=False):
        super(UserVecIterator, self).__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.user_csr_matrix = dataset.to_csr_matrix()
        all_users = np.arange(dataset.num_users, dtype=np.int32)
        self.user_iter = BatchIterator(all_users, batch_size=self.batch_size,
                                       shuffle=self.shuffle, drop_last=self.drop_last)

    def __len__(self):
        return len(self.user_iter)

    def __iter__(self):
        for bat_users in self.user_iter:
            yield self.user_csr_matrix[bat_users].toarray()


class ItemVecIterator(_Iterator):
    def __init__(self, dataset: ImplicitFeedback, batch_size: int=1024,
                 shuffle: bool=True, drop_last: bool=False):
        super(ItemVecIterator, self).__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.item_csr_matrix = dataset.to_csr_matrix().transpose().tocsr()
        all_items = np.arange(dataset.num_items, dtype=np.int32)
        self.item_iter = BatchIterator(all_items, batch_size=self.batch_size,
                                       shuffle=self.shuffle, drop_last=self.drop_last)

    def __len__(self):
        return len(self.item_iter)

    def __iter__(self):
        for bat_items in self.item_iter:
            yield self.item_csr_matrix[bat_items].toarray()


def _generate_positive_triples(head_pos_dict: Dict[int, Dict[str, np.ndarray]]):
    # positive: (h1,r1,t1),  negative (h1,r1,t2)
    assert head_pos_dict, "'head_pos_dict' cannot be empty."

    list_heads, list_relations, list_tails = [], [], []
    head_n_pos = OrderedDict()

    for head, rel_tail_dict in head_pos_dict.items():
        relations = rel_tail_dict[_RELATION]
        tails = rel_tail_dict[_TAIL]
        list_tails.append(tails)
        list_relations.append(relations)
        list_heads.append(np.full_like(tails, head))
        head_n_pos[head] = len(tails)
    heads_ary = np.concatenate(list_heads, axis=0)
    relations_ary = np.concatenate(list_relations, axis=0)
    tails_ary = np.concatenate(list_tails, axis=0)
    return head_n_pos, heads_ary, relations_ary, tails_ary


def _sampling_negative_tails(head_n_pos: OrderedDict, num_neg: int, num_entities: int,
                             head_pos_dict: Dict[int, Dict[str, np.ndarray]]):
    # positive: (h1,r1,t1),  negative (h1,r1,t2)
    assert num_neg > 0, "'num_neg' must be a positive integer."

    neg_tails_list = []
    for head, n_pos in head_n_pos.items():
        neg_tails = randint_choice(num_entities, size=n_pos*num_neg, exclusion=head_pos_dict[head][_TAIL])
        if num_neg == 1:
            neg_tails = neg_tails if isinstance(neg_tails, Iterable) else np.int32([neg_tails])
        else:
            neg_tails = np.reshape(neg_tails, newshape=[n_pos, num_neg])
        neg_tails_list.append(neg_tails)

    return np.concatenate(neg_tails_list)


class KGPairwiseIterator(_Iterator):
    def __init__(self, dataset: KnowledgeGraph, num_neg: int = 1, batch_size: int = 1024,
                 shuffle: bool = True, drop_last: bool = False):
        super(KGPairwiseIterator, self).__init__()
        # positive: (h1,r1,t1),  negative (h1,r1,t2)
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_entities = dataset.num_entities
        self.drop_last = drop_last
        self.head_pos_dict = dataset.to_head_dict()

        self.head_n_pos, self.all_heads, self.relations, self.pos_tails = \
            _generate_positive_triples(self.head_pos_dict)

    def __len__(self):
        n_sample = len(self.all_heads)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        neg_tails = _sampling_negative_tails(self.head_n_pos, self.num_neg,
                                             self.num_entities, self.head_pos_dict)

        data_iter = BatchIterator(self.all_heads, self.relations, self.pos_tails, neg_tails,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_heads, bat_relations, bat_pos_tails, bat_neg_tails in data_iter:
            yield np.asarray(bat_heads), np.asarray(bat_relations), \
                  np.asarray(bat_pos_tails), np.asarray(bat_neg_tails)
