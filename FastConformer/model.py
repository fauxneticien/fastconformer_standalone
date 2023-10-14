import contextlib
import math
import torch

from collections import OrderedDict

from .filterbank_features import FilterbankFeatures
from .conv_subsampling import ConvSubsampling
from .multi_head_attention import RelPositionalEncoding, LocalAttRelPositionalEncoding
from .conformer_encoder import _create_masks, ConformerLayer
from .conv_decoder import ConvASRDecoder

class FastConformer(torch.nn.Module):

    def __init__(
            self,
            num_labels,
            num_layers=18,
            self_attention_model='rel_pos',
            global_tokens=1,
            global_tokens_spacing=1,
            global_attn_separate=False,
            subsampling='dw_striding',
            subsampling_factor=8,
            subsampling_conv_channels=256,
            conv_kernel_size=9,
            conv_context_size=None,
            d_model=512,
            n_heads=8,
            ff_expansion_factor=4,
            att_context_size=[-1,-1],
            dropout=0.1,
            dropout_pre_encoder=0.1,
            dropout_att=0.1,
            dropout_emb=0.1
        ):

        super().__init__()

        self.featurizer = FilterbankFeatures(
            n_window_size=400,
            n_fft=512,
            nfilt=80,
            pad_to=0
        )

        self.pre_encode = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=80,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            subsampling_conv_chunking_factor=1,
            activation=torch.nn.ReLU(),
            is_causal=False,
        )

        self.self_attention_model = self_attention_model

        if self_attention_model == 'rel_pos':
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                xscale=math.sqrt(d_model)
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=d_model,
                dropout_rate=dropout,
                xscale=math.sqrt(d_model),
                dropout_rate_emb=dropout_emb
            )
        else:
            raise ValueError(f"self_attention_model value must be 'rel_pos' or 'rel_pos_local_attn', not '{self_attention_model}'")

        self.layers = torch.nn.ModuleList()
        self.layer_drop_probs = [ 0.0 for _ in range(num_layers) ]

        for i in range(num_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_model * ff_expansion_factor,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_context_size=conv_context_size,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                att_context_size=att_context_size,
                dropout=dropout,
                dropout_att=dropout_att
            )
            self.layers.append(layer)

        self.decoder = ConvASRDecoder(
            feat_in=d_model,
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

    def forward(self, audio_padded, audio_lens):

        fbank_feats, fbank_lens = self.featurizer(audio_padded, audio_lens)

        preenc_feats, preenc_lens = self.pre_encode(
            fbank_feats.transpose(1,2),
            fbank_lens
        )

        max_audio_length=preenc_feats.size(1)
        self.pos_enc.extend_pe(max_audio_length, preenc_feats.device)

        preenc_feats, pos_emb = self.pos_enc(preenc_feats)

        pad_mask, att_mask = self._create_masks(
            att_context_size=[-1, -1] if self.self_attention_model == 'rel_pos' else [128,128],
            padding_length=preenc_lens,
            max_audio_length=max_audio_length,
            offset=None,
            device=preenc_feats.device,
            self_attention_model=self.self_attention_model
        )

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
