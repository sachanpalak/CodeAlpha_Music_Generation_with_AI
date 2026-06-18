import numpy as np
import random
from music21 import note, stream
from sklearn.ensemble import RandomForestClassifier

initial_melody = [
    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
    'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4',
    'C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4',
    'D4', 'F4', 'A4', 'D5', 'A4', 'F4', 'D4'
]

all_notes = initial_melody * 25
unique_pitches = sorted(list(set(all_notes)))

note_to_id = {note_name: i for i, note_name in enumerate(unique_pitches)}
id_to_note = {i: note_name for i, note_name in enumerate(unique_pitches)}

X = []
y = []
sequence_length = 5

for i in range(0, len(all_notes) - sequence_length):
    input_sequence = all_notes[i : i + sequence_length]
    output_note = all_notes[i + sequence_length]
    X.append([note_to_id[char] for char in input_sequence])
    y.append(note_to_id[output_note])

ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X, y)

start_index = random.randint(0, len(X) - 1)
current_pattern = X[start_index]
generated_note_ids = []

for i in range(50):
    model_input = np.array(current_pattern).reshape(1, -1)
    predicted_id = ml_model.predict(model_input)[0]
    generated_note_ids.append(predicted_id)
    current_pattern.append(predicted_id)
    current_pattern = current_pattern[1:]

output_stream = stream.Stream()
note_time_offset = 0.0

for note_id in generated_note_ids:
    note_name = id_to_note[note_id]
    new_note = note.Note(note_name)
    new_note.offset = note_time_offset
    new_note.quarterLength = 1.0
    output_stream.append(new_note)
    note_time_offset += 0.5

output_stream.write('midi', fp='BCA_AI_Music_Output.mid')
print("Success: BCA_AI_Music_Output.mid generated!")