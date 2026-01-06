"""
Face Recognition Module
Handles face encoding and recognition using face_recognition library and LBPH
Enhanced with multi-encoding support and face quality validation for accuracy
"""

import cv2
import numpy as np
import face_recognition
import pickle
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    FACE_RECOGNITION_SETTINGS, TRAINED_MODELS_DIR, DATASET_DIR,
    ATTENDANCE_SETTINGS
)

logger = logging.getLogger(__name__)


class FaceQualityValidator:
    """Validates face image quality for better training accuracy"""

    @staticmethod
    def check_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
        """
        Check if image is blurry using Laplacian variance
        Returns: (is_good_quality, blur_score)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var >= threshold, laplacian_var

    @staticmethod
    def check_brightness(image: np.ndarray, min_brightness: int = 40, max_brightness: int = 220) -> Tuple[bool, float]:
        """
        Check if image has good brightness (not too dark or too bright)
        Returns: (is_good_quality, brightness_value)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        brightness = np.mean(gray)
        is_good = min_brightness <= brightness <= max_brightness
        return is_good, brightness

    @staticmethod
    def check_face_size(face_location: Tuple, min_size: int = None) -> Tuple[bool, int]:
        """
        Check if face is large enough for reliable encoding
        face_location: (top, right, bottom, left) format
        Returns: (is_good_quality, face_size)
        """
        if min_size is None:
            min_size = FACE_RECOGNITION_SETTINGS.get('min_face_size', 80)

        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        face_size = min(face_width, face_height)
        return face_size >= min_size, face_size

    @staticmethod
    def check_face_centered(image_shape: Tuple, face_location: Tuple, margin_ratio: float = 0.1) -> Tuple[bool, dict]:
        """
        Check if face is reasonably centered and not cut off at edges
        Returns: (is_good_quality, position_info)
        """
        height, width = image_shape[:2]
        top, right, bottom, left = face_location

        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)

        is_good = (left >= margin_x and right <= width - margin_x and
                   top >= margin_y and bottom <= height - margin_y)

        position_info = {
            'left_margin': left,
            'right_margin': width - right,
            'top_margin': top,
            'bottom_margin': height - bottom
        }
        return is_good, position_info

    @classmethod
    def validate_face_image(cls, image: np.ndarray, face_location: Tuple = None) -> Tuple[bool, dict]:
        """
        Run all quality checks on a face image
        Returns: (is_valid, quality_report)
        """
        quality_report = {
            'blur': {'passed': False, 'score': 0},
            'brightness': {'passed': False, 'value': 0},
            'face_size': {'passed': False, 'size': 0},
            'centered': {'passed': False, 'info': {}},
            'overall': False
        }

        # Check blur
        blur_ok, blur_score = cls.check_blur(image)
        quality_report['blur'] = {'passed': blur_ok, 'score': blur_score}

        # Check brightness
        bright_ok, brightness = cls.check_brightness(image)
        quality_report['brightness'] = {'passed': bright_ok, 'value': brightness}

        # Check face size and position if face_location provided
        if face_location:
            size_ok, size = cls.check_face_size(face_location)
            quality_report['face_size'] = {'passed': size_ok, 'size': size}

            center_ok, pos_info = cls.check_face_centered(image.shape, face_location)
            quality_report['centered'] = {'passed': center_ok, 'info': pos_info}
        else:
            quality_report['face_size']['passed'] = True
            quality_report['centered']['passed'] = True

        # Overall pass requires blur and brightness at minimum
        quality_report['overall'] = blur_ok and bright_ok

        return quality_report['overall'], quality_report


class FaceRecognizer:
    """
    Face recognition using face_recognition library (dlib-based)
    Enhanced with multi-encoding support for better accuracy
    """

    def __init__(self):
        self.known_encodings = []  # List of encodings
        self.known_ids = []  # Corresponding student IDs
        self.known_names = []  # Corresponding names
        # Multi-encoding storage: {student_id: [list of encodings]}
        self.student_encodings: Dict[str, List[np.ndarray]] = {}
        self.student_names: Dict[str, str] = {}
        self.model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"
        self.quality_validator = FaceQualityValidator()
        self.load_model()

    def get_face_encoding(self, image: np.ndarray,
                          known_locations: list = None) -> Optional[np.ndarray]:
        """Get face encoding from an image"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Get face locations if not provided
            if known_locations is None:
                face_locations = face_recognition.face_locations(
                    rgb_image, model=FACE_RECOGNITION_SETTINGS['model']
                )
            else:
                face_locations = known_locations

            if not face_locations:
                return None

            # Get face encodings
            encodings = face_recognition.face_encodings(
                rgb_image, face_locations,
                num_jitters=FACE_RECOGNITION_SETTINGS['num_jitters'],
                model=FACE_RECOGNITION_SETTINGS['encoding_model']
            )

            return encodings[0] if encodings else None

        except Exception as e:
            logger.error(f"Error getting face encoding: {str(e)}")
            return None

    def get_all_face_encodings(self, image: np.ndarray) -> Tuple[list, list]:
        """Get all face encodings and locations from an image"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(
                rgb_image, model=FACE_RECOGNITION_SETTINGS['model']
            )

            if not face_locations:
                return [], []

            encodings = face_recognition.face_encodings(
                rgb_image, face_locations,
                num_jitters=FACE_RECOGNITION_SETTINGS['num_jitters']
            )

            # Convert locations to (x, y, w, h) format
            face_rects = []
            for (top, right, bottom, left) in face_locations:
                face_rects.append((left, top, right - left, bottom - top))

            return encodings, face_rects

        except Exception as e:
            logger.error(f"Error getting all face encodings: {str(e)}")
            return [], []

    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, str, float]:
        """
        Recognize a face from its encoding using voting across multiple stored encodings
        Returns: (student_id, name, confidence)
        """
        if not self.student_encodings:
            return "Unknown", "Unknown", 0.0

        try:
            tolerance = FACE_RECOGNITION_SETTINGS['tolerance']
            best_match_id = None
            best_match_name = None
            best_confidence = 0.0
            match_votes: Dict[str, List[float]] = {}  # {student_id: [confidences]}

            # Compare against all encodings for each student
            for student_id, encodings in self.student_encodings.items():
                if not encodings:
                    continue

                # Calculate distances to all encodings for this student
                distances = face_recognition.face_distance(encodings, face_encoding)

                # Get matches within tolerance
                matches_within_tolerance = distances[distances <= tolerance]

                if len(matches_within_tolerance) > 0:
                    # Calculate average confidence for matches
                    avg_confidence = 1 - np.mean(matches_within_tolerance)
                    match_count = len(matches_within_tolerance)
                    total_encodings = len(encodings)

                    # Weighted confidence: combines match quality with match ratio
                    # Higher weight if more encodings match
                    match_ratio = match_count / total_encodings
                    weighted_confidence = avg_confidence * (0.7 + 0.3 * match_ratio)

                    match_votes[student_id] = {
                        'confidence': weighted_confidence,
                        'match_count': match_count,
                        'total': total_encodings,
                        'best_distance': np.min(distances)
                    }

            if not match_votes:
                return "Unknown", "Unknown", 0.0

            # Find the best match based on weighted confidence
            best_student_id = max(match_votes.keys(),
                                   key=lambda x: match_votes[x]['confidence'])
            best_match_info = match_votes[best_student_id]

            # Require at least 2 matching encodings OR very high confidence on single match
            min_required = FACE_RECOGNITION_SETTINGS.get('min_encodings_per_student', 5)
            if best_match_info['match_count'] < 2 and best_match_info['confidence'] < 0.7:
                logger.warning(f"Low confidence match rejected: {best_match_info}")
                return "Unknown", "Unknown", best_match_info['confidence']

            return (
                best_student_id,
                self.student_names.get(best_student_id, "Unknown"),
                best_match_info['confidence']
            )

        except Exception as e:
            logger.error(f"Error recognizing face: {str(e)}")
            return "Unknown", "Unknown", 0.0

    def train_model(self, student_data: List[dict]) -> Tuple[bool, str]:
        """
        Train the recognition model with student data
        Stores multiple encodings per student for better accuracy
        student_data: List of {'student_id': str, 'name': str, 'images_path': Path}
        """
        try:
            self.student_encodings = {}
            self.student_names = {}
            # Also maintain flat lists for backward compatibility
            self.known_encodings = []
            self.known_ids = []
            self.known_names = []

            total_images = 0
            quality_rejected = 0
            students_trained = 0

            for student in student_data:
                student_id = student['student_id']
                name = student['name']
                images_path = Path(student['images_path'])

                if not images_path.exists():
                    logger.warning(f"No images found for {student_id}")
                    continue

                # Process all images for this student
                student_valid_encodings = []

                # Support both jpg and png images
                image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + list(images_path.glob('*.jpeg'))

                for img_path in image_files:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    # Validate image quality
                    is_quality_ok, quality_report = self.quality_validator.validate_face_image(image)
                    if not is_quality_ok:
                        quality_rejected += 1
                        logger.debug(f"Image rejected for quality: {img_path} - {quality_report}")
                        continue

                    # Get face location for additional quality check
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(
                        rgb_image, model=FACE_RECOGNITION_SETTINGS['model']
                    )

                    if not face_locations:
                        continue

                    # Check face size
                    size_ok, _ = self.quality_validator.check_face_size(face_locations[0])
                    if not size_ok:
                        quality_rejected += 1
                        continue

                    # Get encoding with high num_jitters for accuracy
                    encodings = face_recognition.face_encodings(
                        rgb_image, face_locations,
                        num_jitters=FACE_RECOGNITION_SETTINGS['num_jitters'],
                        model=FACE_RECOGNITION_SETTINGS['encoding_model']
                    )

                    if encodings:
                        student_valid_encodings.append(encodings[0])
                        total_images += 1

                # Store multiple encodings for this student (don't average)
                min_required = FACE_RECOGNITION_SETTINGS.get('min_encodings_per_student', 5)
                if len(student_valid_encodings) >= min_required:
                    self.student_encodings[student_id] = student_valid_encodings
                    self.student_names[student_id] = name
                    students_trained += 1

                    # Also store average for backward compatibility
                    avg_encoding = np.mean(student_valid_encodings, axis=0)
                    self.known_encodings.append(avg_encoding)
                    self.known_ids.append(student_id)
                    self.known_names.append(name)

                    logger.info(f"Processed {len(student_valid_encodings)} quality images for {name}")
                else:
                    logger.warning(f"Student {name} has only {len(student_valid_encodings)} valid images (need {min_required})")

            # Save the model
            self.save_model()

            msg = f"Training complete: {students_trained} students, {total_images} quality images ({quality_rejected} rejected for quality)"
            logger.info(msg)
            return True, msg

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False, f"Training failed: {str(e)}"

    def train_from_uploaded_images(self, student_id: str, name: str, images: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Train model with uploaded images for a single student
        images: List of BGR numpy arrays
        """
        try:
            valid_encodings = []
            quality_rejected = 0

            for image in images:
                # Validate quality
                is_quality_ok, quality_report = self.quality_validator.validate_face_image(image)
                if not is_quality_ok:
                    quality_rejected += 1
                    continue

                # Get face location
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(
                    rgb_image, model=FACE_RECOGNITION_SETTINGS['model']
                )

                if not face_locations:
                    quality_rejected += 1
                    continue

                # Check face size
                size_ok, _ = self.quality_validator.check_face_size(face_locations[0])
                if not size_ok:
                    quality_rejected += 1
                    continue

                # Get encoding
                encodings = face_recognition.face_encodings(
                    rgb_image, face_locations,
                    num_jitters=FACE_RECOGNITION_SETTINGS['num_jitters'],
                    model=FACE_RECOGNITION_SETTINGS['encoding_model']
                )

                if encodings:
                    valid_encodings.append(encodings[0])

            min_required = FACE_RECOGNITION_SETTINGS.get('min_encodings_per_student', 5)
            if len(valid_encodings) < min_required:
                return False, f"Only {len(valid_encodings)} valid images. Need at least {min_required} quality face images."

            # Add to existing model or create new
            self.student_encodings[student_id] = valid_encodings
            self.student_names[student_id] = name

            # Update flat lists
            if student_id in self.known_ids:
                idx = self.known_ids.index(student_id)
                self.known_encodings[idx] = np.mean(valid_encodings, axis=0)
            else:
                self.known_encodings.append(np.mean(valid_encodings, axis=0))
                self.known_ids.append(student_id)
                self.known_names.append(name)

            self.save_model()

            return True, f"Successfully trained with {len(valid_encodings)} images ({quality_rejected} rejected for quality)"

        except Exception as e:
            logger.error(f"Error training from uploaded images: {str(e)}")
            return False, f"Training failed: {str(e)}"

    def save_model(self):
        """Save trained model to file with multi-encoding support"""
        try:
            TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Convert numpy arrays to lists for JSON serialization
            student_encodings_serializable = {}
            for student_id, encodings in self.student_encodings.items():
                student_encodings_serializable[student_id] = [enc.tolist() for enc in encodings]

            data = {
                'version': 2,  # New format version with multi-encoding
                'student_encodings': student_encodings_serializable,
                'student_names': self.student_names,
                # Keep legacy format for backward compatibility
                'encodings': [enc.tolist() for enc in self.known_encodings],
                'ids': self.known_ids,
                'names': self.known_names
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Model saved to {self.model_path} with {len(self.student_encodings)} students")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load trained model from file with multi-encoding support"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)

                # Check for new format (version 2)
                if data.get('version', 1) >= 2:
                    # Load multi-encoding format
                    self.student_encodings = {}
                    for student_id, encodings in data.get('student_encodings', {}).items():
                        self.student_encodings[student_id] = [np.array(enc) for enc in encodings]
                    self.student_names = data.get('student_names', {})

                # Load legacy format for backward compatibility
                self.known_encodings = [np.array(enc) for enc in data.get('encodings', [])]
                self.known_ids = data.get('ids', [])
                self.known_names = data.get('names', [])

                # If only legacy format exists, convert to new format
                if not self.student_encodings and self.known_ids:
                    for i, student_id in enumerate(self.known_ids):
                        self.student_encodings[student_id] = [self.known_encodings[i]]
                        self.student_names[student_id] = self.known_names[i]

                logger.info(f"Model loaded: {len(self.student_encodings)} students")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")


class LBPHRecognizer:
    """LBPH Face Recognizer for lightweight offline recognition"""

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100
        )
        self.label_map = {}  # Maps numeric labels to student IDs
        self.name_map = {}   # Maps numeric labels to names
        self.model_path = TRAINED_MODELS_DIR / "lbph_model.yml"
        self.label_path = TRAINED_MODELS_DIR / "label_map.json"
        self.load_model()

    def train_model(self, student_data: List[dict]) -> Tuple[bool, str]:
        """Train LBPH model with student images"""
        try:
            faces = []
            labels = []
            label_counter = 0
            total_images = 0

            self.label_map = {}
            self.name_map = {}

            for student in student_data:
                student_id = student['student_id']
                name = student['name']
                images_path = Path(student['images_path'])

                if not images_path.exists():
                    continue

                self.label_map[label_counter] = student_id
                self.name_map[label_counter] = name

                for img_path in images_path.glob('*.jpg'):
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    # Resize for consistency
                    image = cv2.resize(image, (200, 200))
                    faces.append(image)
                    labels.append(label_counter)
                    total_images += 1

                label_counter += 1

            if not faces:
                return False, "No valid images found for training"

            # Train the model
            self.recognizer.train(faces, np.array(labels))
            self.save_model()

            msg = f"LBPH Training complete: {label_counter} students, {total_images} images"
            logger.info(msg)
            return True, msg

        except Exception as e:
            logger.error(f"Error training LBPH: {str(e)}")
            return False, f"Training failed: {str(e)}"

    def recognize_face(self, face_gray: np.ndarray) -> Tuple[str, str, float]:
        """Recognize a face using LBPH"""
        try:
            if not self.label_map:
                return "Unknown", "Unknown", 0.0

            # Resize to expected size
            face_resized = cv2.resize(face_gray, (200, 200))

            # Predict
            label, confidence = self.recognizer.predict(face_resized)

            # Convert LBPH distance to confidence (lower is better)
            # LBPH returns distance, not confidence
            normalized_confidence = max(0, 1 - (confidence / 200))

            if confidence < 100:  # Good match
                student_id = self.label_map.get(label, "Unknown")
                name = self.name_map.get(label, "Unknown")
                return student_id, name, normalized_confidence

            return "Unknown", "Unknown", normalized_confidence

        except Exception as e:
            logger.error(f"Error in LBPH recognition: {str(e)}")
            return "Unknown", "Unknown", 0.0

    def save_model(self):
        """Save LBPH model and label maps"""
        try:
            TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.recognizer.save(str(self.model_path))

            label_data = {
                'label_map': {str(k): v for k, v in self.label_map.items()},
                'name_map': {str(k): v for k, v in self.name_map.items()}
            }
            with open(self.label_path, 'w') as f:
                json.dump(label_data, f)

            logger.info("LBPH model saved")
        except Exception as e:
            logger.error(f"Error saving LBPH model: {str(e)}")

    def load_model(self):
        """Load LBPH model and label maps"""
        try:
            if self.model_path.exists() and self.label_path.exists():
                self.recognizer.read(str(self.model_path))

                with open(self.label_path, 'r') as f:
                    label_data = json.load(f)

                self.label_map = {int(k): v for k, v in label_data['label_map'].items()}
                self.name_map = {int(k): v for k, v in label_data['name_map'].items()}
                logger.info(f"LBPH model loaded: {len(self.label_map)} students")
        except Exception as e:
            logger.error(f"Error loading LBPH model: {str(e)}")


class HybridRecognizer:
    """Combines multiple recognition methods for better accuracy"""

    def __init__(self, use_lbph: bool = True, use_dlib: bool = True):
        self.lbph = LBPHRecognizer() if use_lbph else None
        self.dlib = FaceRecognizer() if use_dlib else None

    def recognize_face(self, image: np.ndarray,
                       face_gray: np.ndarray = None) -> Tuple[str, str, float]:
        """Recognize using multiple methods and combine results"""
        results = []

        # Dlib-based recognition
        if self.dlib:
            encoding = self.dlib.get_face_encoding(image)
            if encoding is not None:
                student_id, name, conf = self.dlib.recognize_face(encoding)
                if student_id != "Unknown":
                    results.append((student_id, name, conf, 'dlib'))

        # LBPH recognition
        if self.lbph and face_gray is not None:
            student_id, name, conf = self.lbph.recognize_face(face_gray)
            if student_id != "Unknown":
                results.append((student_id, name, conf, 'lbph'))

        if not results:
            return "Unknown", "Unknown", 0.0

        # Use dlib result if available (more accurate)
        for r in results:
            if r[3] == 'dlib':
                return r[0], r[1], r[2]

        # Otherwise use best confidence
        best = max(results, key=lambda x: x[2])
        return best[0], best[1], best[2]

    def train_all(self, student_data: List[dict]) -> Tuple[bool, str]:
        """Train all recognizers"""
        messages = []

        if self.dlib:
            success, msg = self.dlib.train_model(student_data)
            messages.append(f"Dlib: {msg}")

        if self.lbph:
            success, msg = self.lbph.train_model(student_data)
            messages.append(f"LBPH: {msg}")

        return True, " | ".join(messages)
