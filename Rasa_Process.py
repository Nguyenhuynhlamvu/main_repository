import os, zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import IPython
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
from gtts import gTTS
import requests
from playsound import playsound
from Global_Variables import *





# Variables
audio_file = 'recorded_audio.wav'




relative_path = '/home/xuanai/Desktop/Library_robot/'
#STT
# load model and tokenizer
model_stt_dir = relative_path + 'models/nguyenvulebinh--wav2vec2-base-vietnamese-250h/snapshots/69e9000591623e5a4fc2f502407860bcdc0de0b2'
processor = Wav2Vec2Processor.from_pretrained(model_stt_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_stt_dir)

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.unk_token_id] = ""
    # convert space character representation
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,
                                   language_model=LanguageModel(lm_model))
    return decoder

lm_file = relative_path + 'models/vi_lm_4grams.bin'
ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_file)

# define function to read in sound file
def map_to_array(batch):
    speech, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch
# speech to text function
def speech2text(audio_file):
  # load dummy dataset and read soundfiles
  ds = map_to_array({"file": audio_file})
  # infer model
  input_values = processor(
        ds["speech"],
        sampling_rate=ds["sampling_rate"],
        return_tensors="pt"
  ).input_values
  # ).input_values.to("cuda")
  # model.to("cuda")
  logits = model(input_values).logits[0]
  # decode ctc output
  pred_ids = torch.argmax(logits, dim=-1)
  greedy_search_output = processor.decode(pred_ids)
  # beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
  return greedy_search_output


def record_audio(file_path, duration=5, sample_rate=16000, chunk=1024, format=pyaudio.paInt16, channels=1):
    # os.system("pyaudio.PyAudio() 2>/dev/null")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    print("..... Recording .....", end='\r')
    frames = []

    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print(" " * 50, end="\r", flush=True)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

#TTS
def text2speech(text):
  # Language in which you want to convert
  language = 'vi'
  # Passing the text and language to the engine, here we have specified slow=False
  tts = gTTS(text=text, lang=language, slow=False)
  # Save the converted audio to a file
  tts.save("output.mp3")
  playsound('output.mp3')

#RASA
def get_rasa_response(user_query, IP_address):
    rasa_server_url = f"http://{IP_address}:5005/webhooks/rest/webhook"
    response = requests.post(rasa_server_url, json={"message": user_query})
    rasa_response = response.json()
    return rasa_response


