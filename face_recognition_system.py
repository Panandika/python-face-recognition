import cv2
import face_recognition
import numpy as np
from typing import List, Tuple
import os
from datetime import datetime


class FaceRecognitionSystem:
    """
    A class to handle face recognition operations using webcam input.
    """
    
    def __init__(self, known_faces_dir: str = "known_faces"):
        """
        Initialize the face recognition system.
        
        Args:
            known_faces_dir (str): Directory containing known face images
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = known_faces_dir
        self.frame_resizing = 0.25  # Resize frame for faster processing
        
        # Create directory for known faces if it doesn't exist
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created directory: {known_faces_dir}")
            print("Please add known face images to this directory")
        
        self.load_known_faces()

    def load_known_faces(self) -> None:
        """
        Load and encode known faces from the specified directory.
        """
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.known_faces_dir, filename)
                try:
                    # Load image and compute encoding
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    
                    # Store encoding and name (filename without extension)
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(os.path.splitext(filename)[0])
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Process a single frame and recognize faces.
        
        Args:
            frame (np.ndarray): Input frame from video stream
            
        Returns:
            Tuple[np.ndarray, List[str]]: Processed frame and list of detected names
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        detected_names = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare face with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                # Use the first matched face
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            detected_names.append(name)
            
            # Scale back face locations
            top, right, bottom, left = [int(coord / self.frame_resizing) for coord in face_location]
            
            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, detected_names

    def run_recognition(self) -> None:
        """
        Run the face recognition system using webcam input.
        """
        if not self.known_face_encodings:
            print("No known faces found. Please add images to the known_faces directory.")
            return
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open video capture device")
            return
        
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame and get results
                processed_frame, detected_names = self.process_frame(frame)
                
                # Display the resulting frame
                cv2.imshow('Face Recognition', processed_frame)
                
                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            video_capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create and run face recognition system
    face_recognition_system = FaceRecognitionSystem()
    face_recognition_system.run_recognition()