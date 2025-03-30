# Biometric Authentication System

This Python-based biometric authentication system combines facial and iris recognition to provide secure user verification. It supports user enrollment, verification, listing of enrolled users, model parameter optimization via a Cuckoo Search algorithm, and deletion of users from the database.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Enrollment Process](#enrollment-process)
  - [Verification Process](#verification-process)
  - [Listing Enrolled Users](#listing-enrolled-users)
  - [Optimizing the Model](#optimizing-the-model)
  - [Deleting a User](#deleting-a-user)
- [Project Structure](#project-structure)
- [Logging](#logging)
- [Our Result](#our-result)
- [License](#license)

## Features

- **User Enrollment:** Extracts facial and iris features from an input image and saves them into a database.
- **User Verification:** Compares the biometric features of a new image against stored templates to verify a user’s identity.
- **Parameter Optimization:** Uses a Cuckoo Search algorithm to optimize system parameters (e.g., Haar cascade scaling factors, iris detection thresholds, and similarity threshold).
- **Command-Line Interface (CLI):** Easily manage the system with commands for enrollment, verification, listing users, optimization, and deletion.

## Requirements

- Python 3.6 or higher
- [OpenCV](https://opencv.org/) (for image processing and face/iris detection)
- [scikit-learn](https://scikit-learn.org/) (for cosine similarity calculation)
- Other standard libraries: `numpy`, `pickle`, `logging`, `argparse`

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/biometric-auth-system.git
   cd biometric-auth-system
2. **Install dependencies:**

   ```bash
   pip install opencv-python opencv-contrib-python numpy scikit-learn

## Commands to Run

1. **Basic Usage**
   ```bash
   python main.py [command] [arguments]
2. **Enrollment Process**
   ```bash
   python main.py enroll --user_id john_doe --image /path/to/face_image.jpg
3. **Verification Process**
   ```bash
   python main.py verify --user_id john_doe --image /path/to/verification_image.jpg
4. **Listing Enrolled Users**
   ```bash
   python main.py list-users
5. **Optimizing the Model**
   Optimize system parameters using a directory of training images. The training directory should have the following structure:
 ```
   training_dir/
 ├── user1/
 │    ├── image1.jpg
 │    └── image2.jpg
 └── user2/
      └── image1.jpg
 ```    
Run the following command to start optimization:
  ```
    python main.py optimize --training_dir /path/to/training_images --epochs 100 --batch_size 32
  ```
6. **Deleting a User**
   Remove an enrolled user from the database:
  ```
    python main.py delete-user --user_id john_doe
  ```

## Project structure
1. **main.py:**
   Contains the core classes and functions:

- BiometricDatabase – Handles the saving and retrieval of biometric templates.
- FeatureExtractor – Detects faces, eyes, and extracts facial and iris features.
- CuckooSearchOptimizer – Implements the Cuckoo Search algorithm for parameter optimization.
- BiometricAuthenticator – Provides enrollment, verification, and parameter optimization functionality.
- Command-line interface to manage user enrollment, verification, optimization, listing, and deletion.

2. **biometric_db.pkl:**
   (Automatically created) Database file that stores enrolled users' biometric templates.

3. **biometric_auth.log:**
   Log file where system operations and errors are recorded.

## Our Result
Below is an image placeholder representing our authentication system results
![WhatsApp Image 2025-03-30 at 12 02 09_23466497](https://github.com/user-attachments/assets/a154fbe4-f595-45e4-bc74-c68a25f3ef58)


