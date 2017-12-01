
# coding: utf-8

# # Sample Melody Generator
#

# In[2]:


import os
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.melody_rnn import melody_rnn_sequence_generator

import magenta
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

import magenta.music as mm


# ### Importing MIDI file
#

# In[6]:


file = magenta.music.midi_io.midi_file_to_sequence_proto("./untitled.mid")


# ### Bundles

# In[24]:


melody = ["basic_rnn", "lookback_rnn", "attention_rnn"]
performance = [
    "performance",
    "performance_with_dynamics",
    "density_conditioned_performance_with_dynamics",
    "pitch_conditioned_performance_with_dynamics",
    "multiconditioned_performance_with_dynamics",
    "optional_multiconditioned_performance_with_dynamics",
]

my_rnn = ['attention_rnn','classical_attention_rnn', 'rock_attention_rnn']

# Constants.
#BUNDLE_DIR = r'./b'
MODEL_NAME = my_rnn[0]
BUNDLE_NAME = MODEL_NAME + '.mag'

# Download bundle .mag file
#mm.notebook_utils.download_bundle(BUNDLE_NAME, BUNDLE_DIR)


# In[14]:


bundle = mm.sequence_generator_bundle.read_bundle_file(BUNDLE_NAME)

# Performance RNN
#generator_per = performance_sequence_generator.get_generator_map()

# Melody RNN
generator_mel = melody_rnn_sequence_generator.get_generator_map()

generator = generator_mel[MODEL_NAME](checkpoint=None, bundle=bundle)
generator.initialize()


generator_options = generator_pb2.GeneratorOptions()
generator_options.args['temperature'].float_value = 1  # Higher is more random; 1.0 is default.
generate_section = generator_options.generate_sections.add(start_time=8, end_time=30)


# In[18]:


# Generate from scratch
#sequence = generator.generate(music_pb2.NoteSequence(), generator_options)


# In[19]:


# Generate from file
sequence = generator.generate(file, generator_options)


# In[20]:


# Play and view this masterpiece.
import pdb;pdb.set_trace()
mm.sequence_proto_to_midi_file(sequence, 'test2.mid')
#mm.plot_sequence(sequence)
#mm.play_sequence(sequence)

