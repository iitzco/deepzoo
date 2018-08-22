import os

from deepspeech.model import Model


class SpeechToText():

    def __init__(self, model_path):
        # Defined constants. See https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py
        BEAM_WIDTH = 500
        LM_WEIGHT = 1.75
        WORD_COUNT_WEIGHT = 1.00
        VALID_WORD_COUNT_WEIGHT = 1.00
        N_FEATURES = 26
        N_CONTEXT = 9

        model = os.path.join(model_path, "output_graph.pb")
        alphabet = os.path.join(model_path, "alphabet.txt")
        lm = os.path.join(model_path, "lm.binary")
        trie = os.path.join(model_path, "trie")

        self.model = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
        self.model.enableDecoderWithLM(alphabet, lm, trie, LM_WEIGHT, WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)

    def run(self, audio, fs):
        return self.model.stt(audio, fs)


