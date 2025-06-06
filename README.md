# Automated Alignment

This repository contains custom modifications made to the original [RBOT (Region-based Object Tracking) implementation](https://github.com/henningtjaden/RBOT). The original code and its documentation are the work of H. Tjaden, U. Schwanecke, E. Schömer, and D. Cremers. This codebase is **not an independent implementation**, but rather an adaptation of their work to suit specific needs.

## About RBOT

RBOT is a novel approach to real-time 6DOF pose estimation of rigid 3D objects using a monocular RGB camera. For a detailed description of the algorithm, dependencies, dataset, and original usage instructions, please refer to the [original README.md](./README.md) in this repository.

## Custom Modifications

This version includes changes and adaptations made for specific requirements.

## Running the Modified Application

### Prerequisites
1. Download the required data from the provided [Google Drive](https://drive.google.com/drive/folders/1pb76Q8hh8D-aLDCxX6_LuIeM71ShlHg5).
2. Download the DLLs from the same drive link and place them in the `app` folder.
2. Ensure you have the necessary 3D models for tracking

### Running RBOT.exe

The `RBOT.exe` file is located in the `app` folder. To run the application:

1. Open Command Prompt (cmd)
2. Navigate to the app directory:
   ```
   cd path\to\app
   ```
3. Run the executable with the following parameters:
   ```
   RBOT.exe [video_path] [model_path] [output_video_path] [frames_folder_path] [save_video]
   ```

#### Command Line Parameters
- `video_path`: Path to the input video file containing the 3D model to be tracked
- `model_path`: Path to the 3D model file (.obj file)
- `output_video_path`: Path where the tracking overlay video will be saved (required if save_video is true)
- `frames_folder_path`: Path to the folder where individual frames will be saved when 'c' is pressed
- `save_video`: Boolean value (true/false) indicating whether to save the tracking overlay video

Example command:
```
RBOT.exe ../data/Motor/Motor1.mp4 ../data/Motor/MotorWithDecimation.obj ../data/Motor/MotorTracking3.mp4 ../data/Motor/ true
```

## Notes
- For building, dependencies, and further usage, see the [original README.md](./README.md).
- This file documents only the customizations and usage relevant to this modified version.
- Please cite the original authors and papers if you use this work in your research.

## License
This project remains under the GNU General Public License Version 3 (GPLv3), as per the original RBOT implementation. 