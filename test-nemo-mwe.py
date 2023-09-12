import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from("tmp/stt_en_fastconformer_ctc_large.nemo")

transcriptions = asr_model.transcribe(['tmp/cat.wav'])

print(transcriptions[0])
