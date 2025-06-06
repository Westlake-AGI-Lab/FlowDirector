# app.py
import gradio as gr
import subprocess
import os
import sys
import datetime
import shutil
import time # Moved import time to the top for global access
import argparse
import cv2
import numpy as np
import tempfile

# --- Configuration ---
# !!! IMPORTANT: Ensure this path is correct for your environment !!!
CKPT_DIR = "./checkpoints/Wan2.1-T2V-1.3B"
EDIT_SCRIPT_PATH = "edit.py"  # Assumes edit.py is in the same directory
OUTPUT_DIR = "gradio_outputs"
PYTHON_EXECUTABLE = sys.executable # Uses the same python that runs gradio
VIDEO_EXAMPLES_DIR = "video_list" # Directory for example videos

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_EXAMPLES_DIR, exist_ok=True) # Ensure video_list exists for clarity

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/Wan2.1-T2V-1.3B",
        help="The path to the checkpoint directory.")

    return parser.parse_args()

def generate_safe_filename_part(text, max_len=20):
    """Generates a filesystem-safe string from text."""
    if not text:
        return "untitled"
    safe_text = "".join(c if c.isalnum() or c in [' ', '_'] else '_' for c in text).strip()
    safe_text = "_".join(safe_text.split()) # Replace spaces with underscores
    return safe_text[:max_len]

def preprocess_video(input_video_path, target_size=(832, 480)):
    """
    Preprocess video to ensure frame count is 4n+1 and resize to target resolution.
    Returns: preprocessed video path, original resolution, and original frame count
    """
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_resolution = (original_width, original_height)
    
    print(f"Original video: {original_width}x{original_height}, {frame_count} frames, {fps} fps")
    
    # Calculate frames to keep (4n+1)
    # Find the largest 4n+1 that is <= frame_count
    frames_to_keep = frame_count
    while (frames_to_keep - 1) % 4 != 0:
        frames_to_keep -= 1
    
    frames_to_drop = frame_count - frames_to_keep
    print(f"Adjusting frame count from {frame_count} to {frames_to_keep} (dropping {frames_to_drop} frames)")
    
    # Create temporary file for preprocessed video
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, target_size)
    
    # Process frames
    frame_idx = 0
    while frame_idx < frames_to_keep:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        out.write(resized_frame)
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    return temp_path, original_resolution, frame_count

def postprocess_video(input_video_path, output_path, original_resolution):
    """
    Postprocess video to resize back to original resolution.
    """
    # Open the edited video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resizing edited video back to original resolution: {original_resolution[0]}x{original_resolution[1]}")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, original_resolution)
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame back to original resolution
        resized_frame = cv2.resize(frame, original_resolution, interpolation=cv2.INTER_AREA)
        out.write(resized_frame)
    
    # Release resources
    cap.release()
    out.release()

def run_video_edit(source_video_path, source_prompt, target_prompt, source_words, target_words, 
                   omega_value, n_max_value, n_avg_value, progress=gr.Progress(track_tqdm=True)):
    if not source_video_path:
        raise gr.Error("Please upload a source video.")
    if not source_prompt:
        raise gr.Error("Please provide a source prompt.")
    if not target_prompt:
        raise gr.Error("Please provide a target prompt (the 'prompt' for edit.py).")
    # Allow empty source_words for additive edits
    if source_words is None: # Check for None, as empty string is valid
         raise gr.Error("Please provide source words (can be empty string for additions).")
    if not target_words:
        raise gr.Error("Please provide target words.")

    preprocessed_video = None
    temp_output_path = None
    
    try:
        progress(0, desc="Preprocessing video...")
        print(f"Source video received at: {source_video_path}")
        
        # Preprocess video: adjust frames to 4n+1 and resize to 832x480
        preprocessed_video, original_resolution, original_frame_count = preprocess_video(source_video_path)
        print(f"Preprocessed video saved to: {preprocessed_video}")
        
        progress(0.05, desc="Preparing for video editing...")
        print(f"Omega value: {omega_value}")
        print(f"N_max value: {n_max_value}")
        print(f"N_avg value: {n_avg_value}")

        worse_avg_value = n_avg_value // 2
        print(f"Calculated Worse_avg value: {worse_avg_value}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        src_words_fn = generate_safe_filename_part(source_words)
        tar_words_fn = generate_safe_filename_part(target_words)
        
        output_filename_base = f"{timestamp}_{src_words_fn}_to_{tar_words_fn}_omega{omega_value}_nmax{n_max_value}_navg{n_avg_value}"
        final_output_path = os.path.join(OUTPUT_DIR, f"{output_filename_base}.mp4")
        
        # Create a temporary output path for the edited video before postprocessing
        temp_fd, temp_output_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)

        cmd = [
            PYTHON_EXECUTABLE, EDIT_SCRIPT_PATH,
            "--task", "t2v-1.3B",
            "--size", "832*480",
            "--base_seed", "42",
            "--ckpt_dir", CKPT_DIR,
            "--sample_solver", "unipc",
            "--source_video_path", preprocessed_video,  # Use preprocessed video
            "--source_prompt", source_prompt,
            "--source_words", source_words, # Pass as is, even if empty
            "--prompt", target_prompt,
            "--target_words", target_words,
            "--sample_guide_scale", "3.5",
            "--tar_guide_scale", "10.5",
            "--sample_shift", "12",
            "--sample_steps", "50",
            "--n_max", str(n_max_value),
            "--n_min", "0", 
            "--n_avg", str(n_avg_value),
            "--worse_avg", str(worse_avg_value), 
            "--omega", str(omega_value),
            "--window_size", "11",
            "--decay_factor", "0.25",
            "--frame_num", "41",
            "--save_file", temp_output_path  # Save to temp path
        ]

        print(f"Executing command: {' '.join(cmd)}")
        progress(0.1, desc="Starting video editing process...")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        
        # Simulate progress
        for i in range(10): 
            if process.poll() is not None: 
                break
            progress(0.1 + i * 0.08, desc=f"Editing in progress... (simulated step {i+1}/10)") 
            time.sleep(1) 

        stdout, stderr = process.communicate() 

        if process.returncode != 0:
            print(f"Error during video editing:\nStdout:\n{stdout}\nStderr:\n{stderr}")
            raise gr.Error(f"Video editing failed. Stderr: {stderr[:500]}")
        
        print(f"Video editing successful. Temp output at: {temp_output_path}")
        if not os.path.exists(temp_output_path):
            print(f"Error: Output file {temp_output_path} was not created.")
            raise gr.Error(f"Output file not found, though script reported success. Stdout: {stdout}")
        
        # Postprocess video: resize back to original resolution
        progress(0.95, desc="Postprocessing video...")
        postprocess_video(temp_output_path, final_output_path, original_resolution)
        
        progress(1, desc="Video ready!")
        return final_output_path

    except FileNotFoundError:
        progress(1, desc="Error")
        print(f"Error: The script '{EDIT_SCRIPT_PATH}' or python executable '{PYTHON_EXECUTABLE}' was not found.")
        raise gr.Error(f"Execution error: Ensure '{EDIT_SCRIPT_PATH}' and Python are correctly pathed.")
    except Exception as e:
        progress(1, desc="Error")
        print(f"An unexpected error occurred: {e}")
        raise gr.Error(f"An unexpected error: {str(e)}")
    finally:
        # Clean up temporary files
        if preprocessed_video and os.path.exists(preprocessed_video):
            try:
                os.remove(preprocessed_video)
                print(f"Cleaned up preprocessed video: {preprocessed_video}")
            except:
                pass
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
                print(f"Cleaned up temp output: {temp_output_path}")
            except:
                pass

# --- Gradio UI Definition ---

# Define all examples to be loaded
examples_to_load_definitions = [
    { # Original bear_g example (corresponds to bear_g_03 in YAML)
        "video_base_name": "bear_g",
        "src_prompt": "A large brown bear is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the bear's deliberate movements.",
        "tar_prompt": "A large dinosaur is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the dinosaur's deliberate movements.",
        "src_words": "large brown bear",
        "tar_words": "large dinosaur",
    },
    { # blackswan_02
        "video_base_name": "blackswan",
        "src_prompt": "A black swan with a red beak swimming in a river near a wall and bushes.",
        "tar_prompt": "A white duck with a red beak swimming in a river near a wall and bushes.",
        "src_words": "black swan",
        "tar_words": "white duck",
    },
    { # jeep_01
        "video_base_name": "jeep",
        "src_prompt": "A silver jeep driving down a curvy road in the countryside.",
        "tar_prompt": "A Porsche car driving down a curvy road in the countryside.",
        "src_words": "silver jeep",
        "tar_words": "Porsche car",
    },
    { # woman_02 (additive edit)
        "video_base_name": "woman",
        "src_prompt": "A woman in a black dress is walking along a paved path in a lush green park, with trees and a wooden bench in the background. The camera remains fixed, capturing her steady movement.",
        "tar_prompt": "A woman in a black dress and a red baseball cap is walking along a paved path in a lush green park, with trees and a wooden bench in the background. The camera remains fixed, capturing her steady movement.",
        "src_words": "", # Empty source words for addition
        "tar_words": "a red baseball cap",
    }
]

examples_data = []
# Default advanced parameters for all examples
default_omega = 2.75
default_n_max = 40
default_n_avg = 4

for ex_def in examples_to_load_definitions:
    # Assuming .mp4 extension for all videos
    video_file_name = f"{ex_def['video_base_name']}.mp4" 
    example_video_path = os.path.join(VIDEO_EXAMPLES_DIR, video_file_name)
    
    if os.path.exists(example_video_path):
        examples_data.append([
            example_video_path,
            ex_def["src_prompt"],
            ex_def["tar_prompt"],
            ex_def["src_words"],
            ex_def["tar_words"],
            default_omega,
            default_n_max,
            default_n_avg
        ])
    else:
        print(f"Warning: Example video {example_video_path} not found. Example for '{ex_def['video_base_name']}' will be skipped.")

if not examples_data:
    print(f"Warning: No example videos found in '{VIDEO_EXAMPLES_DIR}'. Examples section will be empty or not show.")


with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 1400px !important; margin: auto !important;}") as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; font-size: 2.5em;">🪄 FlowDirector Video Edit</h1>
        <p style="text-align: center;">
        Edit videos by providing a source video, descriptive prompts, and specifying words to change.<br>
        Powered by FlowDirector.
        </p>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):  # Input column - increased scale for better width utilization
            with gr.Group():
                gr.Markdown("### 🎬 Source Material")
                source_video_input = gr.Video(label="Upload Source Video", height=480)
                source_prompt_input = gr.Textbox(
                    label="Source Prompt",
                    placeholder="Describe the original video content accurately.",
                    lines=3,
                    show_label=True
                )
                target_prompt_input = gr.Textbox(
                    label="Target Prompt (Desired Edit)",
                    placeholder="Describe how you want the video to be after editing.",
                    lines=3,
                    show_label=True
                )
            
            with gr.Group():
                gr.Markdown("### ✍️ Editing Instructions")
                source_words_input = gr.Textbox(
                    label="Source Words (to be replaced, or empty for addition)",
                    placeholder="e.g., large brown bear (leave empty to add target words globally)"
                )
                target_words_input = gr.Textbox(
                    label="Target Words (replacement or addition)",
                    placeholder="e.g., large dinosaur OR a red baseball cap"
                )

            with gr.Accordion("🔧 Advanced Parameters", open=False):
                omega_slider = gr.Slider(
                    minimum=0.0, maximum=5.0, step=0.05, value=default_omega, label="Omega (ω)",
                    info="Controls the intensity/style of the edit. Higher values might lead to stronger edits."
                )
                n_max_slider = gr.Slider(
                    minimum=0, maximum=50, step=1, value=default_n_max, label="N_max",
                    info="Max value for an adaptive param. `n_min` is fixed at 0."
                )
                n_avg_slider = gr.Slider(
                    minimum=0, maximum=5, step=1, value=default_n_avg, label="N_avg",
                    info="Average value for an adaptive param. `worse_avg` will be N_avg // 2."
                )

            submit_button = gr.Button("✨ Generate Edited Video", variant="primary")

        with gr.Column(scale=2):  # Output column - increased scale for better balance
            gr.Markdown("### 🖼️ Edited Video Output")
            output_video = gr.Video(label="Result", height=480, show_label=False)


    if examples_data: # Only show examples if some were successfully loaded
        gr.Examples(
            examples=examples_data,
            inputs=[
                source_video_input,
                source_prompt_input,
                target_prompt_input,
                source_words_input,
                target_words_input,
                omega_slider,
                n_max_slider, 
                n_avg_slider  
            ],
            outputs=output_video,
            fn=run_video_edit,
            cache_examples=False # For long processes, False is better
        )

    all_process_inputs = [
        source_video_input,
        source_prompt_input,
        target_prompt_input,
        source_words_input,
        target_words_input,
        omega_slider,
        n_max_slider, 
        n_avg_slider  
    ]

    submit_button.click(
        fn=run_video_edit,
        inputs=all_process_inputs,
        outputs=output_video
    )

if __name__ == "__main__":
    # print(f"Make sure your checkpoint directory is correctly set to: {CKPT_DIR}")
    # print(f"And that '{EDIT_SCRIPT_PATH}' is in the same directory as app.py or correctly pathed.")
    # print(f"Outputs will be saved to: {os.path.abspath(OUTPUT_DIR)}")
    # print(f"Place example videos (e.g., bear_g.mp4, blackswan.mp4, etc.) in: {os.path.abspath(VIDEO_EXAMPLES_DIR)}")
    
    args = _parse_args()
    CKPT_DIR = args.ckpt
    demo.launch()
