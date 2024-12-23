import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from allosaurus.app import read_recognizer
from pydub import AudioSegment
import io
import tempfile
from pathlib import Path
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
import plotly.graph_objects as go
from Bio import pairwise2  # BioPython for alignment
from Bio.pairwise2 import format_alignment



# Initialize recognizer (cached)
@st.cache_data
def LoadRecognizer():
    model_directory = Path(r".").resolve()  # Adjust path if needed
    universal_model_name = "uni2005"
    english_model_name = "eng2102"
    try:
        universal_recognizer = read_recognizer(inference_config_or_name=universal_model_name, alt_model_path=model_directory)
        english_recognizer = read_recognizer(inference_config_or_name=english_model_name, alt_model_path=model_directory)
        return universal_recognizer, english_recognizer
    except Exception as e:
        st.error(f"Error loading recognizers: {e}. Ensure Allosaurus models are available.")
        return None, None # Return None values if loading fails

def AudioPreprocessor(Utterance):
    audio_data = Utterance.getbuffer()
    # Detect file format
    file_type = Utterance.type.lower()
    
    # Convert to wav if not already in wav format
    if file_type != 'audio/wav':
        audio = AudioSegment.from_file(io.BytesIO(audio_data))  # Automatically handles m4a, mp3, etc.
    else:
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))

    st.warning(f"Duration: {round(audio.duration_seconds, 2)} seconds")
    st.warning(f"Channels: {audio.channels}")
    st.warning(f"Sample width: {audio.sample_width} bytes")
    
    # Export processed audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
        temp_wav_file_path = temp_wav_file.name
        audio.export(temp_wav_file_path, format="wav")
        return temp_wav_file_path

def plot_spectrogram(audio_path, title):
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

def extract_mfcc(audio_path, sr=16000, n_mfcc=13, win_len=400, hop_len=160):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win_len, hop_length=hop_len, window='hamming')
    return mfcc.T



def calculate_edit_distance_with_details(ref_aligned_phonemes, test_aligned_phonemes):
    """Calculates edit distance based on aligned phonemes using dictionaries."""

    alignment_data = []
    len_aligned = max(len(ref_aligned_phonemes), len(test_aligned_phonemes))

    for i in range(len_aligned):
        ref_phoneme = f"/{ref_aligned_phonemes[i]}/" if i < len(ref_aligned_phonemes) else ""
        test_phoneme = f"/{test_aligned_phonemes[i]}/" if i < len(test_aligned_phonemes) else ""

        operation = ""
        if ref_phoneme == "//" and test_phoneme != "//":
            operation = "Insertion"
        elif ref_phoneme != "//" and test_phoneme == "//":
            operation = "Deletion"
        elif ref_phoneme != test_phoneme:
            operation = "Substitution"
        elif ref_phoneme == test_phoneme and ref_phoneme!="//" and test_phoneme!="//":
            operation = "Match"
        else:
            continue  # Skip if both are empty (no operation)

        # Use a dictionary for each row
        alignment_data.append({
            "Reference Phoneme": ref_phoneme,
            "Test Phoneme": test_phoneme,
            "Operation": operation
        })

    st.table(alignment_data)  # st.table handles dictionaries directly
    return alignment_data


def calculate_distance(mfcc1, mfcc2):
    """Calculates the Euclidean distance between two MFCC sequences."""
    min_len = min(len(mfcc1), len(mfcc2))
    distance = cdist(mfcc1[:min_len], mfcc2[:min_len], metric='euclidean').diagonal()
    return distance



def align_utterances_dtw(ref_mfcc, test_mfcc):
    """Aligns two MFCC sequences using DTW and returns aligned MFCCs."""
    
    # Compute DTW distance and path
    distance, path = fastdtw(ref_mfcc, test_mfcc, dist=lambda x, y: np.linalg.norm(x - y))

    # Get aligned frame indices (path is a list of tuples)
    ref_aligned_frames = [p[0] for p in path]
    test_aligned_frames = [p[1] for p in path]

    # Extract aligned MFCCs
    ref_aligned_mfcc = ref_mfcc[ref_aligned_frames]
    test_aligned_mfcc = test_mfcc[test_aligned_frames]
    
    return ref_aligned_mfcc, test_aligned_mfcc, path



def extract_phonemes(recognition_output):
    """
    Extract phonemes from the recognition output.
    Args:
        recognition_output (str): The output string from the recognizer.
    Returns:
        list: List of phonemes.
    """
    lines = recognition_output.strip().split("\n")
    phonemes = [line.split()[-1] for line in lines if line.strip()]  # Extract the last token on each line
    return phonemes

def calculate_edit_distance_ops(seqA, seqB):
    """
    Calculate edit distance operations (insertion, deletion, substitution) between two sequences.
    Args:
        seqA (str): The reference sequence.
        seqB (str): The test sequence.
    Returns:
        list of dicts: Each dict represents a phoneme with an operation (insertion, deletion, substitution).
    """
    operations = []
    for a, b in zip(seqA, seqB):
        if a == b:
            operations.append({'phoneme': a, 'operation': 'match'})
        elif a == "-":
            operations.append({'phoneme': b, 'operation': 'insertion'})
        elif b == "-":
            operations.append({'phoneme': a, 'operation': 'deletion'})
        else:
            operations.append({'phoneme': f"{a}->{b}", 'operation': 'substitution'})
    return operations



def align_phonemes(ref_phonemes_with_time, test_phonemes_with_time, path):
    """Aligns phonemes using the provided DTW path and returns aligned phoneme sequences."""
    
    ref_aligned_phonemes = []
    test_aligned_phonemes = []

    time_per_frame = 0.01  # Assuming 10ms per frame

    for ref_frame, test_frame in path:
        ref_time = ref_frame * time_per_frame
        test_time = test_frame * time_per_frame

        # Find corresponding phonemes based on time
        ref_phoneme = ""
        test_phoneme = ""
        for phoneme_data in ref_phonemes_with_time:
            start_time, duration, phoneme = phoneme_data.split()
            start_time = float(start_time)
            end_time = start_time + float(duration)
            if start_time <= ref_time <= end_time:
                ref_phoneme = phoneme
                break

        for phoneme_data in test_phonemes_with_time:
            start_time, duration, phoneme = phoneme_data.split()
            start_time = float(start_time)
            end_time = start_time + float(duration)
            if start_time <= test_time <= end_time:
                test_phoneme = phoneme
                break

        ref_aligned_phonemes.append(ref_phoneme)
        test_aligned_phonemes.append(test_phoneme)

    return ref_aligned_phonemes, test_aligned_phonemes




def plot_distance(distance, threshold, ref_aligned_phonemes, test_aligned_phonemes):
    plt.figure(figsize=(16, 8))  # Increased figure size for better readability
    above_threshold = distance > threshold
    plt.plot(np.arange(len(distance)), distance, label='Distance', color='blue')
    plt.fill_between(np.arange(len(distance)), distance, threshold, where=above_threshold, color='red', alpha=0.5, label='Above Threshold')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')

    n = 5  # Show every 5th phoneme (adjust as needed)
    plt.xticks(np.arange(0, len(distance), n), [ref_aligned_phonemes[i] if i < len(ref_aligned_phonemes) else "" for i in range(0, len(distance), n)], rotation=45, ha="right", fontsize=16) # ha for horizontal alignment

    plt.xlabel('Phoneme (Reference)')
    plt.ylabel('Cosine Distance')
    plt.title('Frame-wise Cosine Distance with Phoneme Alignment')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

def plot_distance_plotly(distance, threshold):
    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=np.arange(len(distance)),
        y=distance,
        mode='lines',
        name='Distance',
        hoverinfo="text" # Show only the custom text
    ))

    fig.add_trace(go.Scatter(
        x=np.arange(len(distance)),
        y=[threshold] * len(distance),
        mode='lines',
        name=f'Threshold ({threshold:.2f})',
        line=dict(dash='dash', color='red'),
    ))

    #Highlight above threshold
    x_above = np.where(np.array(distance) > threshold)[0]
    y_above = np.array(distance)[x_above]
    fig.add_trace(go.Scatter(x=x_above, y=y_above, mode='markers', marker=dict(color='red'), name='Above Threshold', hoverinfo="skip"))

    fig.update_layout(
        title='Frame-wise Euclidean Distance with Phoneme Alignment',
        xaxis_title='Frame Index',
        yaxis_title='Euclidean Distance',
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, len(distance), 5),  # Adjust step as needed
            tickangle=-45
        ),
        hovermode="x unified" # Show hover info for all traces at the same x position
    )
    

    st.plotly_chart(fig)

hide_streamlit_style = """
            <style>
                /* Hide the Streamlit header and menu */
                header {visibility: hidden;}
                /* Optionally, hide the footer */
                .streamlit-footer {display: none;}
                /* Hide your specific div class, replace class name with the one you identified */
                .st-emotion-cache-uf99v8 {display: none;}
            </style>
            """

# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("image.png", width = 200)
    
with col3:
    st.write(' ')
    
# Streamlit interface
st.write("### Don't Escape from Excellence, Dive into AccentCave")
st.divider()


# st.title("AccentCave")
LANGUAGE_ID = "arb"
language = st.selectbox("Choose Language", ["Arabic", "Egyption Arabic", "English"])


ref_method = st.radio("Input Reference Audio?", ["Record", "Upload a .wav file"])
if ref_method == "Record":
    ReferenceUtterance = st.audio_input("Record a Reference Utterance")
elif ref_method == "Upload a .wav file":
    ReferenceUtterance = st.file_uploader("Upload a Reference Utterance", ['wav','m4a'])
# 
if ReferenceUtterance is not None:
    reference_audio_path = AudioPreprocessor(ReferenceUtterance)
    plot_spectrogram(reference_audio_path, "Reference Utterance Spectrogram")

    test_method = st.radio("Input Test Audio?", ["Record", "Upload a .wav file"])
    if test_method == "Record":
        TestUtterance = st.audio_input("Record a Test Utterance")
    elif test_method == "Upload a .wav file":
        TestUtterance = st.file_uploader("Upload a Test Utterance", ['wav','m4a'])

    if TestUtterance is not None:
        test_audio_path = AudioPreprocessor(TestUtterance)
        plot_spectrogram(test_audio_path, "Test Utterance Spectrogram")

        reference_mfcc = extract_mfcc(reference_audio_path)
        test_mfcc = extract_mfcc(test_audio_path)


        universal_recognizer, english_recognizer = LoadRecognizer()
        if universal_recognizer is None or english_recognizer is None:
            st.stop() # Stop execution if recognizers failed to load

        if language == "English":
            LANGUAGE_ID = "eng"
            ref_result = english_recognizer.recognize( reference_audio_path, LANGUAGE_ID, timestamp=True)
            test_result = english_recognizer.recognize( test_audio_path, LANGUAGE_ID, timestamp=True)
        else:
            if language == "Arabic":
                LANGUAGE_ID = "arb"
            elif language == "Egyptian Arabic":
                LANGUAGE_ID = "arz"
            ref_result = universal_recognizer.recognize( reference_audio_path, LANGUAGE_ID, timestamp=True)
            test_result = universal_recognizer.recognize( test_audio_path, LANGUAGE_ID, timestamp=True)


        ref_aligned_mfcc, test_aligned_mfcc,path = align_utterances_dtw(reference_mfcc, test_mfcc)

        if ref_result != "" and test_result != "":

            reference_phonemes = extract_phonemes(ref_result)
            test_phonemes = extract_phonemes(test_result)

            # Convert phoneme lists to space-separated strings for alignment
            reference_string = " ".join(reference_phonemes)
            test_string = " ".join(test_phonemes)

            # Align phonemes using Needleman-Wunsch algorithm
            alignments = pairwise2.align.globalxx(reference_string, test_string)

            # Get the best alignment
            alignment = alignments[0]
            aligned_ref = alignment.seqA
            aligned_test = alignment.seqB

            # Calculate edit distance operations
            edit_operations = calculate_edit_distance_ops(aligned_ref.replace(" ", ""), aligned_test.replace(" ", ""))


            table_data = []
            for i, operation in enumerate(edit_operations):
                table_data.append({
                    'Reference Phoneme': aligned_ref[i],
                    'Test Phoneme': aligned_test[i],
                    'Operation': operation['operation'],
                })

            # Display the table using Streamlit
            st.title('Phoneme Alignment and Edit Distance Operations')
            st.table(table_data)

        else:
            ref_aligned_phonemes = None
            test_aligned_phonemes = None
            st.error("No detected Phonemes")

        # Calculate the distance using the aligned MFCCs
        distance = calculate_distance(ref_aligned_mfcc, test_aligned_mfcc)

        distance = distance/len(path)
        threshold = np.mean(distance) * 1.22

        plot_distance_plotly(distance, threshold)