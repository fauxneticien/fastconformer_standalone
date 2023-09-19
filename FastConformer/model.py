import contextlib
import math
import torch

from collections import OrderedDict

from .filterbank_features import FilterbankFeatures
from .conv_subsampling import ConvSubsampling
from .multi_head_attention import RelPositionalEncoding
from .conformer_encoder import _create_masks, ConformerLayer
from .conv_decoder import ConvASRDecoder

class FastConformer(torch.nn.Module):

    def __init__(self, num_labels, num_layers=18):
        super().__init__()

        self.featurizer = FilterbankFeatures(
            n_window_size=400,
            n_fft=512,
            nfilt=80,
            pad_to=0
        )

        self.pre_encode = ConvSubsampling(
            subsampling='dw_striding',
            subsampling_factor=8,
            feat_in=80,
            feat_out=512,
            conv_channels=256,
            subsampling_conv_chunking_factor=1,
            activation=torch.nn.ReLU(),
            is_causal=False,
        )

        self.pos_enc = RelPositionalEncoding(
            d_model=512,
            dropout_rate=0.1,
            xscale=math.sqrt(512)
        )

        self.layers = torch.nn.ModuleList()
        self.layer_drop_probs = [ 0.0 for _ in range(num_layers) ]

        for i in range(num_layers):
            layer = ConformerLayer(
                d_model=512,
                d_ff=2048,
                n_heads=8,
                conv_kernel_size=9,
                conv_context_size=[4,4]
            )
            self.layers.append(layer)

        self.decoder = ConvASRDecoder(
            feat_in=512,
            num_classes=num_labels
        )

    def _create_masks(self, *args, **kwargs):
        return _create_masks(*args, **kwargs)
    
    def load_pretrained_weights(self, ckpt_path, from_nemo=False):
        pretrained_weights = torch.load(ckpt_path, map_location='cpu')

        if from_nemo:
            import re
            # Map official checkpoint keys to our model class
            pretrained_weights = OrderedDict([
                (re.sub(r"(preprocessor|encoder)\.", '', k), v)
                for (k, v) in pretrained_weights.items()
            ])

        self.load_state_dict(pretrained_weights, strict=True)

    def forward(self, audio_padded, audio_lens, update_only_decoder=True):

        fbank_feats, fbank_lens = self.featurizer(audio_padded, audio_lens)

        preenc_feats, preenc_lens = self.pre_encode(
            fbank_feats.transpose(1,2),
            fbank_lens
        )

        max_audio_length=preenc_feats.size(1)
        self.pos_enc.extend_pe(max_audio_length, preenc_feats.device)

        with contextlib.nullcontext() if update_only_decoder else torch.no_grad():
            preenc_feats, pos_emb = self.pos_enc(preenc_feats)

        pad_mask, att_mask = self._create_masks(
            att_context_size=[-1, -1],
            padding_length=preenc_lens,
            max_audio_length=max_audio_length,
            offset=None,
            device=preenc_feats.device,
        )

        with contextlib.nullcontext() if update_only_decoder else torch.no_grad():
            for i, layer in enumerate(self.layers):
                enc_feats = layer(
                    x=preenc_feats if i == 0 else enc_feats,
                    att_mask=att_mask,
                    pos_emb=pos_emb,
                    pad_mask=pad_mask,
                    cache_last_channel=None,
                    cache_last_time=None,
                )

        decoder_output = self.decoder.decoder_layers(
            enc_feats.transpose(1,2)
        ).transpose(1, 2)

        log_probs = torch.nn.functional.log_softmax(decoder_output, dim=-1)

        return log_probs
