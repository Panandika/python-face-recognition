# Simple Face Recognition System

A clean and efficient implementation of a real-time face recognition system using Python, OpenCV, and the face_recognition library.

## Features

- Real-time face detection and recognition using webcam
- Easy addition of new faces to recognize
- Efficient frame processing with resizing optimization
- Clean object-oriented implementation
- Error handling and edge cases management
- Type hints for better code maintainability

## Requirements

- Python 3.6+
- OpenCV
- face_recognition library
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Panandika/python-face-recognition.git
cd python-face-recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Note: The `face_recognition` library requires `dlib`. On some systems, you might need to install additional dependencies. For Ubuntu/Debian:
```bash
sudo apt-get install cmake
sudo apt-get install python3-dlib
```

## Usage

1. Create a directory for known faces (if not exists):
```bash
mkdir known_faces
```

2. Add face images to the `known_faces` directory:
- Use clear, front-facing photos
- Name the files with the person's name (e.g., "john.jpg")
- Supported formats: JPG, JPEG, PNG
- One face per image recommended

3. Run the face recognition system:
```bash
python face_recognition_system.py
```

4. Press 'q' to quit the application

## Performance Optimizations

- Frame resizing for faster processing
- Efficient face encoding storage
- Minimal memory usage
- Early exit conditions

## Error Handling

- Webcam availability check
- Image loading error handling
- Directory existence verification
- Frame capture error detection

## Contributing

Feel free to submit issues and enhancement requests!