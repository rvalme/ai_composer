rm -r /tmp/beethoven.tfrecord
rm -r /tmp/melody_rnn/sequence_examples
cd magenta/scripts
echo 'Converting Midi Directory to note Sequences'
python convert_dir_to_note_sequences.py --input_dir /home/rvalme/antergos_shared/beethoven/ --output_file /tmp/beethoven.tfrecord # creating notesequences
cd ../models/melody_rnn
echo 'Creating SequenceExamples from Note Sequences'
python melody_rnn_create_dataset.py --config attention_rnn --input /tmp/beethoven.tfrecord --output_dir /tmp/melody_rnn/sequence_examples --eval_ratio .10 #create SequenceExamples
echo 'Training the rnn based off of the SequenceExamples'
#python melody_rnn_train.py --num_training_steps 20000 --config attention_rnn --run_dir /tmp/melody_rnn/logdir/run1 --sequence_example_file /tmp/melody_rnn/sequence_examples/training_melodies.tfrecord --hparams batch_size=64,rnn_layer_sizes=[64,64] --num_training_steps 20000
cd ../../..
