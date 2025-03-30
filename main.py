import os
import cv2
import numpy as np
import pickle
import logging
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("biometric_auth.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BiometricAuth")

class BiometricDatabase:
    """Handles storage and retrieval of biometric templates."""
    
    def __init__(self, db_path='biometric_db.pkl'):
        self.db_path = db_path
        self.database = self._load_database()
        
    def _load_database(self):
        """Load the database from disk if it exists, otherwise create a new one."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                return {}
        return {}
    
    def save_database(self):
        """Save the database to disk."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.database, f)
            logger.info("Database saved successfully")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def store_user_template(self, user_id, face_features, iris_features):
        """Store or update a user's biometric templates."""
        self.database[user_id] = {
            'face_features': face_features,
            'iris_features': iris_features,
            'timestamp': datetime.now().isoformat()
        }
        self.save_database()
        logger.info(f"Stored biometric template for user: {user_id}")
    
    def get_user_template(self, user_id):
        """Retrieve a user's biometric templates."""
        if user_id in self.database:
            return self.database[user_id]
        return None
    
    def list_all_users(self):
        """Return a list of all enrolled user IDs."""
        return list(self.database.keys())


class FeatureExtractor:
    """Extracts biometric features from face and iris images."""
    
    def __init__(self, parameters=None):
        # Default parameters if none provided
        self.parameters = parameters or {
            'face_scale_factor': 1.1,
            'face_min_neighbors': 5,
            'iris_param1': 200,
            'iris_param2': 50,
            'iris_min_radius': 10,
            'iris_max_radius': 50
        }
        
        # Load OpenCV face and eye detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Face recognition model
        # For simplicity, we're using a basic feature extractor, but in a production system
        # you would want to use a deep learning model for face embeddings
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
    def update_parameters(self, parameters):
        """Update the feature extraction parameters."""
        self.parameters.update(parameters)
        logger.info(f"Updated feature extraction parameters: {parameters}")
    
    def detect_face(self, image):
        """Detect a face in the given image and return the face region."""
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_face")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.parameters['face_scale_factor'],
            minNeighbors=self.parameters['face_min_neighbors'],
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            return None
        
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_region = gray[y:y+h, x:x+w]
        
        logger.info(f"Face detected at coordinates: x={x}, y={y}, width={w}, height={h}")
        return face_region, (x, y, w, h)
    
    def detect_iris(self, image, face_rect=None):
        """Detect iris in the given image and return the iris region."""
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_iris")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # If face detection is available, focus on eye regions
        if face_rect:
            x, y, w, h = face_rect
            # Define a region of interest for eyes (upper half of face)
            roi_gray = gray[y:y+h//2, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) == 0:
                logger.warning("No eyes detected in the face region")
                return None, None
            
            # Process the first eye detected
            ex, ey, ew, eh = eyes[0]
            eye_center = (x + ex + ew//2, y + ey + eh//2)
            eye_region = gray[y+ey:y+ey+eh, x+ex:x+ex+ew]
            
            # Find circles in the eye region (iris)
            circles = cv2.HoughCircles(
                eye_region,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=self.parameters['iris_param1'],
                param2=self.parameters['iris_param2'],
                minRadius=self.parameters['iris_min_radius'],
                maxRadius=self.parameters['iris_max_radius']
            )
            
            if circles is not None:
                # Convert circle parameters to integer
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    # Create an iris mask and extract the iris region
                    mask = np.zeros(eye_region.shape, dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), r, 255, -1)
                    iris_region = cv2.bitwise_and(eye_region, eye_region, mask=mask)
                    
                    logger.info(f"Iris detected in eye region at coordinates: x={cx}, y={cy}, radius={r}")
                    return iris_region, (x+ex+cx, y+ey+cy, r)
        
        logger.warning("No iris detected in the image")
        return None, None
    
    def extract_face_features(self, face_image):
        """Extract facial features from the face image."""
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided for feature extraction")
            return None
        
        # Resize for consistent feature extraction
        face_resized = cv2.resize(face_image, (100, 100))
        
        # For a simple feature vector, we'll use raw pixel values and HOG features
        # In a production system, you'd use a deep learning-based face embedding model
        
        # 1. Histogram of Oriented Gradients (HOG) features
        hog = cv2.HOGDescriptor((100, 100), (20, 20), (10, 10), (10, 10), 9)
        hog_features = hog.compute(face_resized)
        
        # 2. Local Binary Patterns (LBP) features
        lbp_features = self._compute_lbp(face_resized)
        
        # Combine the features
        combined_features = np.concatenate([hog_features.flatten(), lbp_features.flatten()])
        
        return combined_features
    
    def _compute_lbp(self, image, radius=1, neighbors=8):
        """Compute Local Binary Pattern features."""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_code = 0
                
                # Sample points in a clockwise direction
                for k in range(neighbors):
                    angle = 2 * math.pi * k / neighbors
                    x = j + radius * math.cos(angle)
                    y = i - radius * math.sin(angle)
                    
                    # Bilinear interpolation
                    fx, cx = math.modf(x)
                    fy, cy = math.modf(y)
                    
                    cx, cy = int(cx), int(cy)
                    
                    # Check boundaries
                    if cx < 0 or cx >= cols - 1 or cy < 0 or cy >= rows - 1:
                        continue
                    
                    # Bilinear interpolation
                    tl = image[cy, cx]
                    tr = image[cy, cx + 1]
                    bl = image[cy + 1, cx]
                    br = image[cy + 1, cx + 1]
                    
                    value = (1 - fx) * (1 - fy) * tl + \
                            fx * (1 - fy) * tr + \
                            (1 - fx) * fy * bl + \
                            fx * fy * br
                    
                    if value >= center:
                        binary_code |= (1 << k)
                
                lbp[i, j] = binary_code
        
        # Compute histogram of LBP values
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 2**neighbors + 1), density=True)
        return hist
    
    def extract_iris_features(self, iris_image):
        """Extract iris features from the iris image."""
        if iris_image is None or iris_image.size == 0:
            logger.warning("Empty iris image provided for feature extraction")
            return None
        
        # Resize for consistent feature extraction
        iris_resized = cv2.resize(iris_image, (50, 50))
        
        # Apply a Gabor filter bank to extract iris texture features
        # In a real-world application, you'd use more sophisticated iris recognition algorithms
        
        features = []
        for theta in np.arange(0, np.pi, np.pi/4):
            for freq in [0.1, 0.2, 0.3]:
                kernel = cv2.getGaborKernel((15, 15), 4.0, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(iris_resized, cv2.CV_8UC3, kernel)
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
        
        # Add histogram features
        hist = cv2.calcHist([iris_resized], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        return np.array(features)


class CuckooSearchOptimizer:
    """Implements Cuckoo Search algorithm for parameter optimization."""
    
    def __init__(self, param_ranges, n_nests=10, pa=0.25, max_generations=20):
        """
        Initialize the Cuckoo Search optimizer.
        
        Args:
            param_ranges: Dictionary of parameter names and their min/max values
            n_nests: Number of nests (solution candidates)
            pa: Probability of egg abandonment
            max_generations: Maximum number of iterations
        """
        self.param_ranges = param_ranges
        self.n_nests = n_nests
        self.pa = pa
        self.max_generations = max_generations
        
        # Initialize random nests
        self.nests = self._initialize_nests()
        self.best_nest = None
        self.best_fitness = -float('inf')
        
    def _initialize_nests(self):
        """Initialize random nests within parameter ranges."""
        nests = []
        for _ in range(self.n_nests):
            nest = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                if isinstance(min_val, float):
                    nest[param] = min_val + random.random() * (max_val - min_val)
                else:
                    nest[param] = random.randint(min_val, max_val)
            nests.append(nest)
        return nests
    
    def _levy_flight(self):
        """Implement Lévy flight."""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, 1)[0]
        v = np.random.normal(0, 1, 1)[0]
        step = u / abs(v) ** (1 / beta)
        return step
    
    def _get_new_solution(self, nest):
        """Generate a new solution via Lévy flight."""
        new_nest = nest.copy()
        
        # Apply Lévy flight to a randomly selected parameter
        param = random.choice(list(self.param_ranges.keys()))
        min_val, max_val = self.param_ranges[param]
        
        # Calculate step size using Lévy flight
        step = self._levy_flight() * 0.01 * (max_val - min_val)
        
        # Update parameter value
        if isinstance(min_val, float):
            new_nest[param] += step
            new_nest[param] = max(min_val, min(max_val, new_nest[param]))
        else:
            step_int = max(1, int(abs(step)))
            new_nest[param] += random.choice([-step_int, step_int])
            new_nest[param] = max(min_val, min(max_val, new_nest[param]))
        
        return new_nest
    
    def _abandon_nests(self, nests, fitness_values):
        """Abandon worst nests and build new ones."""
        # Sort nests by fitness
        sorted_indices = np.argsort(fitness_values)
        
        # Abandon worst nests with probability pa
        for i in range(int(self.pa * self.n_nests)):
            nest_idx = sorted_indices[i]
            nests[nest_idx] = self._initialize_nests()[0]
        
        return nests
    
    def optimize(self, fitness_function):
        """
        Run the Cuckoo Search optimization algorithm.
        
        Args:
            fitness_function: Function to evaluate the fitness of a nest (parameter set)
        
        Returns:
            The best parameter set found and its fitness value
        """
        logger.info("Starting Cuckoo Search optimization")
        
        # Evaluate initial nests
        fitness_values = [fitness_function(nest) for nest in self.nests]
        best_idx = np.argmax(fitness_values)
        self.best_nest = self.nests[best_idx]
        self.best_fitness = fitness_values[best_idx]
        
        # Main optimization loop
        for generation in range(self.max_generations):
            # Get a random nest
            i = random.randrange(self.n_nests)
            
            # Generate a new solution
            new_nest = self._get_new_solution(self.nests[i])
            
            # Evaluate the new solution
            new_fitness = fitness_function(new_nest)
            
            # Replace if better
            if new_fitness > fitness_values[i]:
                self.nests[i] = new_nest
                fitness_values[i] = new_fitness
                
                # Update best solution if needed
                if new_fitness > self.best_fitness:
                    self.best_nest = new_nest
                    self.best_fitness = new_fitness
            
            # Abandon some nests
            self.nests = self._abandon_nests(self.nests, fitness_values)
            
            # Re-evaluate fitness values for abandoned nests
            fitness_values = [fitness_function(nest) for nest in self.nests]
            
            logger.info(f"Generation {generation+1}/{self.max_generations} - Best fitness: {self.best_fitness}")
            logger.info(f"Best parameters: {self.best_nest}")
        
        logger.info(f"Optimization completed - Best fitness: {self.best_fitness}")
        logger.info(f"Optimized parameters: {self.best_nest}")
        
        return self.best_nest, self.best_fitness


class BiometricAuthenticator:
    """Main class for biometric authentication system."""
    
    def __init__(self, db_path='biometric_db.pkl', similarity_threshold=0.7):
        """Initialize the biometric authentication system."""
        self.database = BiometricDatabase(db_path)
        self.feature_extractor = FeatureExtractor()
        self.similarity_threshold = similarity_threshold
        logger.info("Biometric authentication system initialized")
    
    def enroll_user(self, user_id, image_path):
        """
        Enroll a new user or update an existing user's biometric templates.
        
        Args:
            user_id: Unique identifier for the user
            image_path: Path to the image containing the user's face
        
        Returns:
            True if enrollment was successful, False otherwise
        """
        logger.info(f"Enrolling user: {user_id} with image: {image_path}")
        
        try:
            # Load and process the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return False
            
            # Detect face and extract features
            face_result = self.feature_extractor.detect_face(image)
            if face_result is None:
                logger.error(f"No face detected in the image for user: {user_id}")
                return False
            
            face_image, face_rect = face_result
            face_features = self.feature_extractor.extract_face_features(face_image)
            
            # Detect iris and extract features
            iris_result = self.feature_extractor.detect_iris(image, face_rect)
            if iris_result is None or iris_result[0] is None:
                logger.warning(f"No iris detected in the image for user: {user_id}")
                iris_features = np.zeros(100)  # Placeholder for missing iris features
            else:
                iris_image, _ = iris_result
                iris_features = self.feature_extractor.extract_iris_features(iris_image)
            
            # Store the templates
            self.database.store_user_template(user_id, face_features, iris_features)
            logger.info(f"Successfully enrolled user: {user_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during user enrollment: {e}")
            return False
    
    def verify_user(self, user_id, image_path):
        """
        Verify a user's identity using biometric features.
        
        Args:
            user_id: Identifier of the user to verify
            image_path: Path to the image for verification
        
        Returns:
            (is_verified, confidence_score) tuple
        """
        logger.info(f"Verifying user: {user_id} with image: {image_path}")
        
        try:
            # Get the stored template for the user
            template = self.database.get_user_template(user_id)
            if template is None:
                logger.error(f"No template found for user: {user_id}")
                return False, 0.0
            
            # Load and process the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return False, 0.0
            
            # Detect face and extract features
            face_result = self.feature_extractor.detect_face(image)
            if face_result is None:
                logger.error(f"No face detected in the verification image")
                return False, 0.0
            
            face_image, face_rect = face_result
            face_features = self.feature_extractor.extract_face_features(face_image)
            
            # Detect iris and extract features
            iris_result = self.feature_extractor.detect_iris(image, face_rect)
            if iris_result is None or iris_result[0] is None:
                logger.warning(f"No iris detected in the verification image")
                iris_features = np.zeros(100)  # Placeholder for missing iris features
            else:
                iris_image, _ = iris_result
                iris_features = self.feature_extractor.extract_iris_features(iris_image)
            
            # Calculate similarity scores
            face_similarity = self._calculate_similarity(face_features, template['face_features'])
            iris_similarity = self._calculate_similarity(iris_features, template['iris_features'])
            
            # Weighted combination of face and iris scores
            # Adjust weights based on confidence in each modality
            face_weight = 0.7
            iris_weight = 0.3
            
            if iris_features is None or np.all(iris_features == 0):
                # If iris features are missing, rely only on face
                combined_score = face_similarity
            else:
                combined_score = face_weight * face_similarity + iris_weight * iris_similarity
            
            is_verified = combined_score >= self.similarity_threshold
            
            logger.info(f"Verification result for user {user_id}: {'Success' if is_verified else 'Failed'}")
            logger.info(f"Face similarity: {face_similarity:.4f}, Iris similarity: {iris_similarity:.4f}")
            logger.info(f"Combined score: {combined_score:.4f}, Threshold: {self.similarity_threshold:.4f}")
            
            return is_verified, combined_score
        
        except Exception as e:
            logger.error(f"Error during user verification: {e}")
            return False, 0.0
    
    def _calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors."""
        if features1 is None or features2 is None:
            return 0.0
        
        # Reshape features to 2D arrays for cosine_similarity
        features1_2d = features1.reshape(1, -1)
        features2_2d = features2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(features1_2d, features2_2d)[0][0]
        return similarity
    
    def optimize_parameters(self, training_data):
        """
        Optimize the system parameters using Cuckoo Search algorithm.
        
        Args:
            training_data: List of (user_id, image_paths) tuples for training
        
        Returns:
            Optimized parameters
        """
        logger.info("Starting parameter optimization with Cuckoo Search")
        
        # Define parameter ranges for optimization
        param_ranges = {
            'face_scale_factor': (1.05, 1.3),
            'face_min_neighbors': (3, 7),
            'iris_param1': (100, 300),
            'iris_param2': (30, 70),
            'iris_min_radius': (5, 15),
            'iris_max_radius': (30, 60),
            'similarity_threshold': (0.5, 0.9)
        }
        
        # Initialize Cuckoo Search optimizer
        optimizer = CuckooSearchOptimizer(param_ranges, n_nests=10, max_generations=20)
        
        # Define the fitness function for parameter evaluation
        def fitness_function(parameters):
            # Update the feature extractor with the new parameters
            self.feature_extractor.update_parameters(parameters)
            
            # If similarity threshold is included in the parameters, update it
            if 'similarity_threshold' in parameters:
                self.similarity_threshold = parameters['similarity_threshold']
            
            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0
            
            # Evaluate on the training data
            for user_id, image_paths in training_data:
                # Enroll the user with the first image
                enrollment_success = self.enroll_user(user_id, image_paths[0])
                
                if not enrollment_success:
                    continue
                
                # Verify with the remaining images of the same user (should be positive)
                for i in range(1, len(image_paths)):
                    is_verified, _ = self.verify_user(user_id, image_paths[i])
                    if is_verified:
                        true_positives += 1
                    else:
                        false_negatives += 1
                
                # Verify with images of other users (should be negative)
                for other_user, other_images in training_data:
                    if other_user == user_id:
                        continue
                    
                    # Use only the first image of other users for negative testing
                    is_verified, _ = self.verify_user(user_id, other_images[0])
                    if is_verified:
                        false_positives += 1
                    else:
                        true_negatives += 1
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # F1 score as the fitness value
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f"Fitness evaluation - F1 Score: {f1_score:.4f}")
            logger.info(f"TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")
            
            return f1_score
        
        # Run the optimization
        best_params, best_fitness = optimizer.optimize(fitness_function)
        
        # Update the system with the optimized parameters
        self.feature_extractor.update_parameters(best_params)
        if 'similarity_threshold' in best_params:
            self.similarity_threshold = best_params['similarity_threshold']
        
        logger.info(f"Parameter optimization completed - Best F1 Score: {best_fitness:.4f}")
        logger.info(f"Optimized parameters: {best_params}")
        
        return best_params


def create_command_line_interface():
    """Create a command-line interface for the biometric authentication system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Biometric Authentication System')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Enroll command
    enroll_parser = subparsers.add_parser('enroll', help='Enroll a new user')
    enroll_parser.add_argument('--user-id', type=str, required=True, help='User ID')
    enroll_parser.add_argument('--image', type=str, required=True, help='Path to the enrollment image')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify a user')
    verify_parser.add_argument('--user-id', type=str, required=True, help='User ID to verify')
    verify_parser.add_argument('--image', type=str, required=True, help='Path to the verification image')
    
    # List users command
    subparsers.add_parser('list-users', help='List all enrolled users')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize system parameters')
    optimize_parser.add_argument('--training-dir', type=str, required=True, help='Directory containing training images')
    
    return parser


def find_training_data(training_dir):
    """
    Find training data in the specified directory.
    
    Expected directory structure:
    training_dir/
        user1/
            image1.jpg
            image2.jpg
            ...
        user2/
            image1.jpg
            ...
    
    Returns:
        List of (user_id, image_paths) tuples
    """
    training_data = []
    
    # Iterate through user directories
    for user_id in os.listdir(training_dir):
        user_dir = os.path.join(training_dir, user_id)
        
        if not os.path.isdir(user_dir):
            continue
        
        # Find image files for this user
        image_paths = []
        for filename in os.listdir(user_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(user_dir, filename))
        
        if image_paths:
            training_data.append((user_id, image_paths))
    
    return training_data


def main():
    """Main function to run the biometric authentication system."""
    parser = create_command_line_interface()
    args = parser.parse_args()
    
    authenticator = BiometricAuthenticator()
    
    if args.command == 'enroll':
        success = authenticator.enroll_user(args.user_id, args.image)
        if success:
            print(f"Successfully enrolled user: {args.user_id}")
        else:
            print(f"Failed to enroll user: {args.user_id}")
    
    elif args.command == 'verify':
        is_verified, confidence = authenticator.verify_user(args.user_id, args.image)
        if is_verified:
            print(f"User {args.user_id} verified successfully with confidence: {confidence:.4f}")
        else:
            print(f"Verification failed for user {args.user_id}. Confidence: {confidence:.4f}")
    
    elif args.command == 'list-users':
        users = authenticator.database.list_all_users()
        if users:
            print("Enrolled users:")
            for user_id in users:
                print(f"- {user_id}")
        else:
            print("No users enrolled yet.")
    
    elif args.command == 'optimize':
        if not os.path.isdir(args.training_dir):
            print(f"Training directory not found: {args.training_dir}")
            return
        
        training_data = find_training_data(args.training_dir)
        if not training_data:
            print("No training data found in the specified directory.")
            return
            
        print(f"Found {len(training_data)} training samples. Starting optimization...")
        authenticator.optimize_model(training_data, epochs=args.epochs, batch_size=args.batch_size)
        print("Model optimization completed.")
    elif args.command == 'delete-user':
        if authenticator.database.delete_user(args.user_id):
            print(f"User {args.user_id} deleted successfully.")
        else:
            print(f"Failed to delete user {args.user_id}. User may not exist.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()