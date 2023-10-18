import math
import torch

import lightning.pytorch as pl
import numpy as np

from lhotse import CutSet
from lhotse.dataset import (
    DynamicBucketingSampler,
    IterableDatasetWrapper,
    make_worker_init_fn
)
from lhotse.dataset.collation import collate_custom_field
from pathlib import Path
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader

# from ._helpers import LnLTokenCollater

def predict_lengths(input_len, n_downsamples):
    import math
    from functools import reduce
    def pred_single_downsample(input_len, k=3, s=2, p=1):
        output_len = math.floor((input_len+(2*p)-k)/s)+1
        return output_len
    
    return reduce(lambda i, f: f(i), [ pred_single_downsample ] * n_downsamples, input_len)

class FastConformerPretrainingDataset(torch.utils.data.Dataset):

    def __init__(self, frameshift):
        self.frameshift=frameshift

    def _predict_preenc_len(self, fbank_len, k=3, s=2, p=1):
        def _helper(input_len):
            return math.floor((input_len+(2*p)-k)/s)+1
        
        len_20ms = _helper(fbank_len)
        len_40ms = _helper(len_20ms)

        if self.frameshift == 40:
            return len_40ms
        elif self.frameshift == 80:
            return _helper(len_40ms)
        else:
            raise ValueError('Frameshift expected to be 40 or 80')

    def __getitem__(self, cuts: CutSet) -> dict:
        feats_padded, feats_lens = collate_custom_field(cuts, 'fbank', pad_value=0)
        # Note we'll use a negative integer for padding the labels since 0 is a valid label
        # ptlabels_padded, ptlabels_lens = collate_custom_field(cuts, 'ptlabel', pad_value=-100)

        ptlabels = [ torch.IntTensor(c.load_ptlabel()).to(torch.int64) for c in cuts ]
        ptlabels_padded=torch.nn.utils.rnn.pad_sequence(ptlabels, batch_first=True, padding_value=-1)
        ptlabels_lens=torch.IntTensor([ p.shape[0] for p in ptlabels ]).to(torch.int64)

        return {
            "feats_padded": feats_padded,
            "ptlabels_padded": ptlabels_padded,
            "feats_lens": feats_lens,
            "ptlabels_lens": ptlabels_lens
        }

class LibriSpeechDataModule(pl.LightningDataModule):

    def __init__(self,
                 stage,
                 train_part,
                 dev_part,
                 frameshift,
                 train_max_dur=300,
                 dev_max_dur=300,
                 corpus_dir="./data/_shar/LibriSpeech_all",
                 num_dl_workers=1
                 ):
        
        super().__init__()
        self.prepare_data_per_node = False

        assert stage in ["pretrain", "finetune"]

        self.stage = stage
        self.frameshift = frameshift

        self.train_part = train_part
        self.train_max_dur = train_max_dur

        self.dev_part = dev_part
        self.dev_max_dur = dev_max_dur

        self.corpus_dir = Path(corpus_dir)
        self.num_dl_workers = num_dl_workers

    def setup(self, stage = None):

        fields = {}

        for part_name, part_path in [('train', self.train_part), ('dev', self.dev_part)]:
            fields[part_name] = {
                'cuts': sorted(list(self.corpus_dir.glob(f"{part_path}/cuts.*.jsonl.gz"))),
                'fbank': sorted(list(self.corpus_dir.glob(f"{part_path}/fbank_10ms.*.tar"))),
            }

            if self.stage == 'pretrain':
                fields[part_name]['ptlabel'] = sorted(list(self.corpus_dir.glob(f"{part_path}/ptlabel_{self.frameshift}ms.*.tar")))

        self.cuts_train = CutSet.from_shar(
            fields=fields['train'],
            # The three arguments below are specifically for dataloading.
            # shuffle_shards=True enables shuffling of shards,
            # stateful_shuffle=True makes the shuffling different on each epoch,
            # and seed="randomized" tells the CutSet to randomize the seed on each dataloader node and worker.
            shuffle_shards=True,
            stateful_shuffle=True,
            seed="randomized",
        )

        # if self.stage == 'finetune':
        #     pass
        #     self.tokenizer = LnLTokenCollater(self.cuts_train,  add_unk=False, add_bos=False, add_eos=False)
        #     self.tokenizer_n_class = len(list(self.tokenizer.idx2token))

        # Set to repeat only after tokenizer configuration which loops over all supervisions
        # otherwise enters infinite loop!
        self.cuts_train = self.cuts_train.repeat()

        self.cuts_dev = CutSet.from_shar(
            fields=fields['dev'],
            shuffle_shards=False
        )

        self.sampler_train = DynamicBucketingSampler(
            self.cuts_train,
            shuffle=True,
            max_duration=self.train_max_dur,
            drop_last=True,
            num_buckets=10,
            # For the training data, set rank=0 and world_size=1 but move sampler to each worker's process and use an iterable-style dataset (see train_dataloader() below)
            # https://colab.research.google.com/drive/1TdWooKi_u5uJsFENXI-tk-sYRGHvVDAE#scrollTo=Lzgjvf1puSoL
            rank=0,
            world_size=1
        )

        self.sampler_dev = DynamicBucketingSampler(
            self.cuts_dev,
            shuffle=False,
            max_duration=self.dev_max_dur,
            drop_last=True,
            num_buckets=10,
            rank=get_rank(),
            world_size=get_world_size()
            # For the dev data, set rank and world size as usual but keep sampler in main process and use a map-style dataset (see val_dataloader() below)
        )

    def train_dataloader(self):

        return DataLoader(
            IterableDatasetWrapper(
                # Convert map-stype dataset to iterable-style dataset (on infinite repeat, via repeat())
                dataset=FastConformerPretrainingDataset(self.frameshift),
                # This sampler has rank=0, world_size=1 set
                sampler=self.sampler_train
            ),
            batch_size=None,
            num_workers=self.num_dl_workers,
            worker_init_fn=make_worker_init_fn(rank=get_rank(), world_size=get_world_size(), seed=0)
        )
    
    def val_dataloader(self):

        return DataLoader(            
            # This is a map-style dataset (not on infinite repeat)
            dataset=FastConformerPretrainingDataset(self.frameshift),
            # This sampler has rank=X, world_size=Y set as usual
            sampler=self.sampler_dev,
            batch_size=None,
            num_workers=self.num_dl_workers,
        )
