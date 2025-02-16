from flask import Flask, Response, render_template, jsonify, request, session, redirect, url_for, flash
from functools import wraps
import cv2 as cv
import mediapipe as mp
import numpy as np
import serial
import time
import csv
import copy
import itertools
import os
import sqlite3
import logging
import atexit
from datetime import timedelta 
from flask_socketio import SocketIO, emit
import threading
from functools import wraps
from collections import Counter, deque
from utils import CvFpsCalc
from model import KeyPointClassifier
from waitress import serve
from database import GestureDatabase

# Initialize Flask app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'gesture_data.db')


app = Flask(__name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

app.config['SECRET_KEY'] = os.urandom(24)  # Generate secure random key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Initialize SocketIO with correct configuration
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25
)

# logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global variables
camera = None
hands = None
keypoint_classifier = None
gesture_labels = []
point_history = deque(maxlen=16)
arduino = None
db = None
last_gesture = None
last_gesture_time = 0
GESTURE_COOLDOWN = 1.0  # Seconds between gesture logs


def cleanup():
    """Clean up resources on server shutdown"""
    global camera, db
    if camera is not None:
        camera.release()
    if db:
        db.close()
    cv.destroyAllWindows()

atexit.register(cleanup)

# Login decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if 'user_id' not in session:
                flash('Please log in to access this page', 'warning')
                return redirect(url_for('login', next=request.url))
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Login decorator error: {e}")
            flash('An error occurred', 'error')
            return redirect(url_for('login'))
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if not session.get('is_admin'):
                flash('Admin access required', 'error')
                return redirect(url_for('login', next=request.url))
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Admin decorator error: {e}")
            flash('An error occurred', 'error')
            return redirect(url_for('login'))
    return decorated_function

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('Please enter both username and password', 'error')
                return render_template('login.html')
            
            if db:
                user = db.verify_user(username, password)
                if user:
                    # Store username instead of id
                    session['user_id'] = user['username']  # Changed from user['id']
                    session['is_admin'] = user['is_admin']
                    session.permanent = True
                    
                    next_page = request.args.get('next')
                    return redirect(next_page or url_for('index'))
                flash('Invalid username or password', 'error')
            else:
                flash('Database error', 'error')
        
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Login error: {e}")
        flash('An error occurred', 'error')
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def initialize_serial(port='COM9', baud_rate=9600):
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        print(f"Port {port} opened successfully")
        return ser
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def initialize_components():
    global camera, hands, keypoint_classifier, gesture_labels, arduino, db
    
    try:
        # Initialize database
        if not os.path.exists(os.path.dirname(DB_PATH)):
            os.makedirs(os.path.dirname(DB_PATH))
        db = GestureDatabase(DB_PATH)

        if not db.test_connection():
            logger.error("Database connection failed")
            return False
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Use fastest model
        )
        
        # Initialize classifier
        model_path = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
            
        keypoint_classifier = KeyPointClassifier(model_path=model_path)
        
        # Load labels
        label_path = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')
        if not os.path.exists(label_path):
            raise Exception(f"Label file not found: {label_path}")
            
        with open(label_path, encoding='utf-8-sig') as f:
            gesture_labels = [row[0] for row in csv.reader(f)]
            
        # arduino = initialize_serial()
        print("All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera
    try:
        logger.info("Toggle camera request received")
        if camera is None:
            logger.info("Attempting to open camera")
            camera = cv.VideoCapture(0, cv.CAP_DSHOW)
            
            if not camera.isOpened():
                logger.error("Failed to open camera")
                return jsonify({'success': False, 'message': 'Failed to open camera'})
            
            camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv.CAP_PROP_FPS, 30)
            
            logger.info("Camera opened successfully")
            return jsonify({'success': True, 'status': 'opened'})
        else:
            logger.info("Closing camera")
            camera.release()
            camera = None
            cv.destroyAllWindows()  # Add this line
            return jsonify({'success': True, 'status': 'closed'})
    except Exception as e:
        logger.error(f"Camera toggle error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/gesture_stats')
def gesture_stats():
    if db:
        try:
            stats = db.get_gesture_stats()
            return jsonify({
                'success': True,
                'stats': stats
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    return jsonify({'success': False, 'message': 'Database not initialized'})

# Update the generate_frames function

def generate_frames():
    global camera, point_history, last_gesture, last_gesture_time
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        try:
            if camera is None or not camera.isOpened():
                time.sleep(0.1)
                continue
                
            fps = cvFpsCalc.get()
            success, image = camera.read()
            if not success:
                continue

            # Process frame with timestamp handling
            debug_image = cv.flip(image, 1)  # Mirror display
            image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
            
            # To improve MediaPipe performance and handle timestamps
            image.flags.writeable = False
            current_time = int(time.time() * 1000)  # Convert to milliseconds
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    try:
                        # Calculate landmark coordinates
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        
                        # Classify gesture
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        if 0 <= hand_sign_id < len(gesture_labels):
                            gesture_name = gesture_labels[hand_sign_id]
                            current_time_sec = time.time()
                            
                            # Gesture cooldown check with proper time comparison
                            if (gesture_name != last_gesture or 
                                current_time_sec - last_gesture_time >= GESTURE_COOLDOWN):
                                last_gesture = gesture_name
                                last_gesture_time = current_time_sec
                                
                                try:
                                    # Log to database
                                    db.log_gesture(gesture_name)
                                    count = db.get_gesture_count(gesture_name)
                                    
                                    # Prepare gesture data
                                    gesture_data = {
                                        'name': gesture_name,
                                        'timestamp': time.strftime('%H:%M:%S'),
                                        'count': count,
                                        'confidence': float(handedness.classification[0].score)
                                    }
                                    
                                    # Debug logging
                                    logger.debug(f"Emitting gesture: {gesture_data}")
                                    
                                    # Emit using socketio instance
                                    socketio.emit('gesture_detected', gesture_data, namespace='/')
                                    
                                except Exception as e:
                                    logger.error(f"Error handling gesture: {e}")
                                    continue
                                
                        # Draw visualization
                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(debug_image, handedness, gesture_name)
                        
                    except Exception as e:
                        logger.error(f"Hand processing error: {e}")
                        continue

            # Add FPS counter
            cv.putText(debug_image, f"FPS: {int(fps)}", (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(debug_image, f"FPS: {int(fps)}", (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

            # Encode and yield frame
            ret, buffer = cv.imencode('.jpg', debug_image, [int(cv.IMWRITE_JPEG_QUALITY), 85])
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Add small delay to control frame rate
            time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            time.sleep(0.1)
            continue

@app.route('/get_gesture_data')
def get_gesture_data():
    if db:
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT g.name, g.description, COUNT(h.id) as count
                FROM gestures g
                LEFT JOIN gesture_history h ON g.id = h.gesture_id
                GROUP BY g.name, g.description
                ORDER BY g.id
            ''')
            
            gestures = cursor.fetchall()
            return jsonify({
                'success': True,
                'data': [
                    {
                        'name': g[0],
                        'description': g[1],
                        'count': g[2]
                    } for g in gestures
                ]
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
        finally:
            if conn:
                conn.close()
    return jsonify({'success': False, 'message': 'Database not initialized'})

@app.route('/')
def index():
    if session.get('is_admin') and request.path == '/admin':
        return redirect(url_for('admin_panel'))
    return render_template('index.html')

# Add this before the main execution block

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors"""
    try:
        return render_template('404.html'), 404
    except Exception as e:
        logger.error(f"Error handling 404: {e}")
        return "Page not found", 404

@app.route('/admin')
@admin_required
def admin_panel():
    try:
        if db:
            gestures = db.get_all_gestures()
            users = db.get_all_users()
            return render_template('admin.html', gestures=gestures, users=users)
        flash('Database not initialized', 'error')
        logger.error("Database not initialized when accessing admin panel")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Admin panel error: {e}")
        flash('Error loading admin panel', 'error')
        return redirect(url_for('index'))

# Add new admin routes for user management
@app.route('/admin/user', methods=['POST'])
@admin_required
def add_user():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        is_admin = request.form.get('is_admin') == 'true'
        if db and username and password:
            success = db.add_user(username, password, is_admin)
            return jsonify({'success': success})
    return jsonify({'success': False})

@app.route('/admin/user/<int:id>', methods=['PUT', 'DELETE'])
@admin_required
def manage_user(id):
    if request.method == 'PUT':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        is_admin = data.get('is_admin')
        if db and username:
            success = db.update_user(id, username, password, is_admin)
            return jsonify({'success': success})
    elif request.method == 'DELETE':
        if db:
            success = db.delete_user(id)
            return jsonify({'success': success})
    return jsonify({'success': False})

@app.route('/admin/users')
@admin_required
def get_users():
    if db:
        users = db.get_all_users()
        return jsonify({'success': True, 'users': users})
    return jsonify({'success': False, 'message': 'Database not initialized'})

@app.route('/video_feed')
def video_feed():
    try:
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"Video feed error: {e}")
        return "Video feed error", 500
    
# Add these routes after the socketio initialization and before the routes

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info('Client connected')
    emit('connected', {'data': 'Connected successfully'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info('Client disconnected')


@app.route('/debug/socket-test') 
def socket_test():
    """Test WebSocket connection"""
    try:
        test_data = {
            'name': 'TEST_GESTURE',
            'timestamp': time.strftime('%H:%M:%S'),
            'count': 1
        }
        socketio.emit('gesture_detected', test_data)
        return jsonify({'success': True, 'message': 'Test data emitted'})
    except Exception as e:
        logger.error(f"Socket test error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug/emit-test')
def debug_emit_test():
    """Debug route to test WebSocket emission"""
    try:
        test_data = {
            'name': 'TEST_GESTURE',
            'timestamp': time.strftime('%H:%M:%S'),
            'count': 1,
            'confidence': 1.0
        }
        socketio.emit('gesture_detected', test_data)
        logger.info(f"Test data emitted: {test_data}")
        return jsonify({'success': True, 'message': 'Test data emitted'})
    except Exception as e:
        logger.error(f"Emit test error: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/debug/test-gesture-log')
def test_gesture_log():
    """Test route to verify gesture logging system"""
    try:
        # Create test gesture data
        test_gestures = [
            {
                'name': 'TEST_GESTURE_1',
                'timestamp': time.strftime('%H:%M:%S'),
                'count': 1,
                'confidence': 0.95
            },
            {
                'name': 'TEST_GESTURE_2',
                'timestamp': time.strftime('%H:%M:%S'),
                'count': 2,
                'confidence': 0.98
            }
        ]
        
        # Log test gestures
        for gesture in test_gestures:
            socketio.emit('gesture_detected', gesture)
            logger.info(f"Emitted test gesture: {gesture}")
            time.sleep(1)  # Add delay between emissions
            
        return jsonify({
            'success': True, 
            'message': 'Test gestures emitted',
            'gestures': test_gestures
        })
    except Exception as e:
        logger.error(f"Test gesture log error: {e}")
        return jsonify({'success': False, 'error': str(e)})

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

def draw_landmarks(image, landmark_points):
    if len(landmark_points) > 0:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Little
        ]
        
        for connection in connections:
            cv.line(image, tuple(landmark_points[connection[0]]), 
                   tuple(landmark_points[connection[1]]), (0, 255, 0), 2)
        
        for point in landmark_points:
            cv.circle(image, tuple(point), 5, (0, 0, 255), -1)
    
    return image

def draw_info_text(image, handedness, gesture_name):
    info_text = f"{handedness.classification[0].label[0]}: {gesture_name}"
    cv.putText(image, info_text, (10, 60),
              cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, info_text, (10, 60),
              cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

# Update the main execution block at the bottom of the file
if __name__ == '__main__':
    if initialize_components():
        try:
            print("Starting server at http://127.0.0.1:5000")
            # Use socketio.run instead of app.run
            socketio.run(
                app,
                host="127.0.0.1",
                port=5000,
                debug=False,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            print(f"Server error: {e}")
            logger.error(f"Server initialization error: {e}")
        finally:
            cleanup()