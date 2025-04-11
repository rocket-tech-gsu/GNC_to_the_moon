import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import imageio
import argparse
from tqdm import tqdm
import os

def create_video_for_column(x_values, y_values, col_name, output_video, fps=10):
    # Create a blank figure.
    fig = go.Figure()
    
    # Add an empty line trace.
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=col_name))
    
    # Set up layout.
    fig.update_layout(
        title=f"Animated Line Graph for {col_name}",
        xaxis_title=x_values.name,
        yaxis_title=col_name
    )
    
    images = []
    
    # Generate frames: for each row, update the trace with data up to that row.
    for i in tqdm(range(1, len(x_values)+1), desc=f"Generating frames for {col_name}"):
        fig.data[0].x = x_values.iloc[:i]
        fig.data[0].y = y_values.iloc[:i]
        fig.update_layout(width=1280, height=720)
        # Render the frame to a PNG image using Kaleido.
        img_bytes = pio.to_image(fig, format="png", width=1280, height=720)
        image = imageio.v2.imread(img_bytes)
        images.append(image)
    
    # Write frames to a video file.
    writer = imageio.get_writer(output_video, fps=fps)
    for img in tqdm(images, desc=f"Writing video for {col_name}"):
        writer.append_data(img)
    writer.close()
    
    print(f"Video saved as {output_video}")

def create_videos_from_csv(csv_file, output_dir="videos", fps=10):
    # Read CSV into a DataFrame.
    df = pd.read_csv(csv_file)
    
    # Assume the first column is the x-axis.
    x_values = df.iloc[:, 0]
    x_values.name = df.columns[0]
    
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each column starting from index 2.
    for col in df.columns[3:]:
        y_values = df[col]
        output_video = os.path.join(output_dir, f"{col}.mp4")
        create_video_for_column(x_values, y_values, col, output_video, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animated Line Graph Video Exporter for Each Column in CSV")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="videos", help="Directory to save output video files")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the videos")
    args = parser.parse_args()
    create_videos_from_csv(args.csv_file, args.output_dir, args.fps)
