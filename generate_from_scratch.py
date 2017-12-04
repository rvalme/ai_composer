import os
# from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.melody_rnn import melody_rnn_sequence_generator
import magenta
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2
import magenta.music as mm

melody = ["basic_rnn", "lookback_rnn", "attention_rnn"]
performance = [
    "performance",
    "performance_with_dynamics",
    "density_conditioned_performance_with_dynamics",
    "pitch_conditioned_performance_with_dynamics",
    "multiconditioned_performance_with_dynamics",
    "optional_multiconditioned_performance_with_dynamics",
]

my_rnn = ['our/classical/attention_rnn','our/blues/attention_rnn', 'our/rock/attention_rnn']

# Constants.
BUNDLE_DIR = r'./bundles/'
MODEL_BASED_NAME = melody[2]
MODEL_NAME = my_rnn[0]
BUNDLE_NAME = BUNDLE_DIR + MODEL_NAME + '.mag'

bundle = mm.sequence_generator_bundle.read_bundle_file(BUNDLE_NAME)

# Melody RNN
generator_mel = melody_rnn_sequence_generator.get_generator_map()
generator = generator_mel[MODEL_BASED_NAME](checkpoint=None, bundle=bundle)
generator.initialize()


generator_options = generator_pb2.GeneratorOptions()
generator_options.args['temperature'].float_value = 1  # Higher is more random; 1.0 is default.
generate_section = generator_options.generate_sections.add(start_time=0, end_time=20)

sequence = generator.generate(music_pb2.NoteSequence(), generator_options)
mm.sequence_proto_to_midi_file(sequence, 'sample.mid')

# for i in range(20):
    # sequence = generator.generate(music_pb2.NoteSequence(), generator_options)
    # mm.sequence_proto_to_midi_file(sequence, 'sample'+str(i)+'.mid')