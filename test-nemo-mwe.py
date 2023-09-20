import torch
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from("tmp/stt_en_fastconformer_ctc_large.nemo")
asr_model.eval()

with torch.inference_mode():
    transcriptions = asr_model.transcribe(['tmp/cat.wav'])

print(transcriptions[0])

# Change to LongFormer-style local attention
asr_model.change_attention_model(
    self_attention_model="rel_pos_local_attn",
    att_context_size=[128, 128]
)

with torch.inference_mode():
    transcriptions = asr_model.transcribe(['tmp/cat.wav'])

print(transcriptions[0])
