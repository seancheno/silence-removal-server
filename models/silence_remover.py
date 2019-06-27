import keras
from keras import backend as K
from keras.models import model_from_json
import librosa
import numpy as np
from tqdm import tqdm
import os

from aws import s3
from config import s3_bucket_url


class SilenceRemoverModel:
    def __init__(self, data):
        if data.filename != "sampleAudio.wav":
            self.filename = "/var/www/flask/{}".format(data.filename)
            data.save(self.filename)
        else:
            self.filename = "/var/www/flask/sample-audio.wav"
        self.samples, self.sr = librosa.load(
            self.filename, mono=True, sr=16000)
        self.seconds_removed = 0

    def run(self):
        mfcc_window_vectors = self.split_audio_into_mfccs()
        predictions = self.predict_silence(mfcc_window_vectors)
        silence_removed_samples = self.remove_predicted_silence(predictions)

        self.save_as_wav(silence_removed_samples)
        self.upload_to_s3(silence_removed_samples)
        self.delete_wav()

        condenced_audio_samples = self.condence_audio_samples(
            silence_removed_samples)

        return self.create_payload_for_client(condenced_audio_samples)

    def split_audio_into_mfccs(self):
        # Load the wave data and break it up into indiviusal MFCC's as input for the model
        mfcc = librosa.feature.mfcc(self.samples, sr=self.sr)
        mfcc_window_vectors = []
        for i in tqdm(range(0, int(mfcc.shape[1])), "Saving mfcc's"):
            mfcc_window = mfcc[:, i:i+1]
            mfcc_window_vectors.append(mfcc_window)

        mfcc_window_vectors = np.stack(mfcc_window_vectors, axis=0)
        return mfcc_window_vectors

    def predict_silence(self, mfcc_window_vectors):
        # # Model reconstruction from JSON file
        model = None
        with open('/var/www/flask/model.json', 'r') as f:
            model = model_from_json(f.read())

        # Load weights into the new model
        model.load_weights('/var/www/flask/model.h5')

        # Predict with the model on every MFCC in the mfcc_window_vectors
        predictions = np.zeros(shape=(0, 2))

        for i in tqdm(range(0, mfcc_window_vectors.shape[0]), "Predicting with model"):
            current_mfcc = mfcc_window_vectors[i]
            mfcc_reshaped = current_mfcc.reshape(1, 20, 1, 1)

            prob = model.predict_proba(mfcc_reshaped)
            predictions = np.vstack((predictions, prob))

        K.clear_session()

        return predictions

    def remove_predicted_silence(self, predictions):
        # Remove detected silence from audio
        samples_per_mfcc = 500
        samples_silence_removed = np.zeros(shape=(0))
        for index, value in np.ndenumerate(predictions[:, 1]):
            if value < .70:
                start_index = index[0] * samples_per_mfcc
                non_silence_samples = self.samples[start_index: start_index +
                                                   samples_per_mfcc]
                samples_silence_removed = np.append(
                    samples_silence_removed, non_silence_samples)

        return samples_silence_removed

    def save_as_wav(self, samples_silence_removed):
        # Save new file to local storage
        librosa.output.write_wav(
            self.filename, samples_silence_removed, self.sr)
        # samples_silence_removed.export("audio_tested.wav", format="wav")

    def upload_to_s3(self, audio_samples):
        # Upload to s3
        data = open(self.filename, 'rb')
        s3.Bucket('silenceremoval').put_object(
            Key=self.filename, Body=data, ACL='public-read')

    def delete_wav(self):
        if os.path.exists(self.filename):
            if self.filename != "/var/www/flask/sample-audio.wav":
                os.remove(self.filename)

    def condence_audio_samples(self, samples_silence_removed):

        # Condence new and original wav file data
        chunk_length = 1000
        condenced_samples = self.samples.tolist(
        )[::(int(len(self.samples)/chunk_length))]
        condenced_samples_silence_removed = samples_silence_removed.tolist()[
            ::(int(len(samples_silence_removed)/1000))]
        condenced_audio_samples = [
            condenced_samples, condenced_samples_silence_removed]

        self.seconds_removed = (len(self.samples) / 16000) - \
            (len(samples_silence_removed) / 16000)

        print('condenced_samples: {}'.format(len(condenced_samples)))
        print('condenced_samples_silence_removed: {}'.format(
            len(condenced_samples_silence_removed)))
        return condenced_audio_samples

    def create_payload_for_client(self, condenced_audio_samples):
        print('condenced_samples: {}'.format(len(condenced_audio_samples[0])))
        print('condenced_samples_silence_removed: {}'.format(
            len(condenced_audio_samples[1])))

        # Create payload data
        payload = {
            'samples': condenced_audio_samples[0], 'samples_silence_removed': condenced_audio_samples[1], 's3_audio_url': s3_bucket_url + self.filename, 'seconds_removed': self.seconds_removed}

        print(self.seconds_removed)

        return payload
