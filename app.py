from models import prado
import tensorflow as tf
from layers import projection_layers
from layers import base_layers

LABELS = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
    'neutral',
]

MODEL_CONFIG = {
    'labels': LABELS,
    'multilabel': True,
    'quantize': False,
    'max_seq_len': 128,
    'max_seq_len_inference': 128,
    'exclude_nonalphaspace_unicodes': False,
    'split_on_space': True,
    'embedding_regularizer_scale': 0.035,
    'embedding_size': 64,
    'bigram_channels': 64,
    'trigram_channels': 64,
    'feature_size': 512,
    'network_regularizer_scale': 0.0001,
    'keep_prob': 0.5,
    'word_novelty_bits': 0,
    'doc_size_levels': 0,
    'add_bos_tag': False,
    'add_eos_tag': False,
    'pre_logits_fc_layers': [],
    'text_distortion_probability': 0.0,
}

def build_model():

    # Otherwise, we get string inputs which we need to project.
    input = tf.keras.Input(shape=(), name='input', dtype='string')
    projection_layer = projection_layers.ProjectionLayer(MODEL_CONFIG, base_layers.PREDICT)
    projection, sequence_length = projection_layer(input)
    inputs = [input]

    # Next we add the model layer.
    model_layer = prado.Encoder(MODEL_CONFIG, base_layers.PREDICT)
    logits = model_layer(projection, sequence_length)

    # Finally we add an activation layer.
    if MODEL_CONFIG['multilabel']:
        activation = tf.keras.layers.Activation('sigmoid', name='predictions')
    else:
        activation = tf.keras.layers.Activation('softmax', name='predictions')
    predictions = activation(logits)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[predictions])
    
    return model

import numpy as np

EMOJI_MAP = {
    'admiration': '👏',
    'amusement': '😂',
    'anger': '😡',
    'annoyance': '😒',
    'approval': '👍',
    'caring': '🤗',
    'confusion': '😕',
    'curiosity': '🤔',
    'desire': '😍',
    'disappointment': '😞',
    'disapproval': '👎',
    'disgust': '🤮',
    'embarrassment': '😳',
    'excitement': '🤩',
    'fear': '😨',
    'gratitude': '🙏',
    'grief': '😢',
    'joy': '😃',
    'love': '❤️',
    'nervousness': '😬',
    'optimism': '🤞',
    'pride': '😌',
    'realization': '💡',
    'relief': '😅',
    'remorse': '',
    'sadness': '😞',
    'surprise': '😲',
    'neutral': '',
}

PREDICT_TEXT = [
  b'Good for you!',
  b'Happy birthday!',
  b'I love you.',
]

model = build_model(base_layers.PREDICT)
model.load_model('weight/model')

for text in PREDICT_TEXT:
  results = model.predict(x=[text])
  print('')
  print('{}:'.format(text))
  labels = np.flip(np.argsort(results[0]))
  for x in range(3):
    label = LABELS[labels[x]]
    label = EMOJI_MAP[label] if EMOJI_MAP[label] else label
    print('{}: {}'.format(label, results[0][labels[x]]))