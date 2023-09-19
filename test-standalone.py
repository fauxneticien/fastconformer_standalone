import torch
import torchaudio

import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from("tmp/stt_en_fastconformer_ctc_large.nemo")

samples, sample_rate = torchaudio.load("tmp/cat.wav")

audio_samples = samples
audio_lens    = torch.tensor(samples.shape[1]).unsqueeze(0)

print("Using NeMo FastConformer...")

fbank_feats, fbank_lens = asr_model.preprocessor.featurizer(audio_samples.to('cuda'), audio_lens.to('cuda'))

preenc_feats, preenc_lens = asr_model.encoder.pre_encode(
    fbank_feats.transpose(1,2),
    fbank_lens
)

preenc_feats, pos_emb = asr_model.encoder.pos_enc(preenc_feats)

pad_mask, att_mask = asr_model.encoder._create_masks(
    att_context_size=asr_model.encoder.att_context_size,
    padding_length=preenc_lens,
    max_audio_length=preenc_feats.size(1),
    offset=None,
    device=preenc_feats.device,
)

audio_signal = preenc_feats

for lth, (drop_prob, layer) in enumerate(zip(asr_model.encoder.layer_drop_probs, asr_model.encoder.layers)):
    original_signal = audio_signal

    audio_signal = layer(
        x=audio_signal,
        att_mask=att_mask,
        pos_emb=pos_emb,
        pad_mask=pad_mask,
        cache_last_channel=None,
        cache_last_time=None,
    )

decoder_output = asr_model.decoder.decoder_layers(
    audio_signal.transpose(1,2)
).transpose(1, 2)

nemo_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=-1)

nemo_hypotheses, all_hyp = asr_model.decoding.ctc_decoder_predictions_tensor(
    nemo_log_probs,
    decoder_lengths=preenc_lens,
    return_hypotheses=False
)

print(nemo_hypotheses)

print("Using standalone FastConformer...")

from FastConformer.model import FastConformer

fc = FastConformer(num_labels=1024)
fc.load_pretrained_weights("tmp/nemo_unpacked/model_weights.ckpt", from_nemo=True)
fc.to('cuda')

standalone_log_probs = fc(audio_samples.to('cuda'), audio_lens.to('cuda'))
# Just use NeMo's decoder for testing
standalone_hypotheses, all_hyp = asr_model.decoding.ctc_decoder_predictions_tensor(
    standalone_log_probs,
    decoder_lengths=preenc_lens,
    return_hypotheses=False
)

print(standalone_hypotheses)
