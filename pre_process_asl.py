import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

class Preprocess_ASL():
    vid_base_dir = 'video_data'
    videos_per_word = 100
    frames_per_video = 75
    def __init__(self):
        # Creating the holistic model
        self.mp_holistic = mp.solutions.holistic

        self.holistic_model = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # For drawing landmarks over images
        self.mp_draw = mp.solutions.drawing_utils
    
    def word_videos_to_frames(self, start_word=0, end_word=1):
        frame_count = 0
        for word_id in range(start_word, end_word):
            for video_id in range(self.videos_per_word):
                # Create frames directory next to each video
                video_dir = os.path.join(self.vid_base_dir, f'{word_id}', f'{video_id}')
                video_file_path = os.path.join(video_dir, f'{word_id}_{video_id}.mp4')
                frames_dir = os.path.join(video_dir,'frames')
    #            print(frames_dir)
    #            print(video_file_path)
    #            print(video_dir)
                if os.path.exists(frames_dir) is False:
                    #print(f'Creating new frames directory {frames_dir}')
                    os.makedirs(frames_dir)
                else:
                    pass
                    #print(f'{frames_dir} already exists')
                
                # Open video capture of current video
                cap = cv2.VideoCapture(video_file_path)

                frame_num = 0
                while True:
                    # Read next frame
                    ret, frame = cap.read()

                    # Break when video is over
                    if not ret:
                        break

                    # Save frames to frame_file_path
                    frame_file_path = os.path.join(frames_dir, f'{word_id}_{video_id}_{frame_num}.png')
                    #print(f'saving frame {word_id}_{video_id}_{frame_num}.png to {frame_file_path}')
                    cv2.imwrite(frame_file_path, frame)
                    frame_num += 1
                    frame_count += 1
                    
                cap.release()
        return frame_count
    
    def open_image_detect_holistic(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Load the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the color to RGB to pass into the holistic model
        img.flags.writeable = False # Avoid any bugs by not allowing the image to be modified
        landmarks = self.holistic_model.process(img) # Extract landmarks from the image
        img.flags.writeable = True  # Make the image writeable again
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Conver the color back to the original BGR format

        return img, landmarks
    
    def draw_over_image(self, image, landmarks):
        #Draw Face
        self.mp_draw.draw_landmarks(image, landmarks.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, 
                            self.mp_draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                            self.mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
        
        #Draw pose
        self.mp_draw.draw_landmarks(image, landmarks.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2))
        
        #Draw left hand
        self.mp_draw.draw_landmarks(image, landmarks.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=2))
        
        #Draw right hand
        self.mp_draw.draw_landmarks(image, landmarks.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(127,127,127), thickness=1, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=2))
        
        plt.imshow(image)
        plt.show()

    def retrieve_landmarks(self, landmark_results):
        frame_pose_lm = []
        frame_face_lm = []
        frame_left_hand_lm = []
        frame_right_hand_lm = []
        
        # Get landmarks for pose
        if landmark_results.pose_landmarks is None:
            frame_pose_lm = np.zeros((33,4))
        else:
            for lm in landmark_results.pose_landmarks.landmark:
                pose_lm = np.array([lm.x, lm.y, lm.z, lm.visibility])
                frame_pose_lm.append(pose_lm)

        # Get landmarks for face
        if landmark_results.face_landmarks is None:
            frame_face_lm = np.zeros((468,3))
        else:
            for lm in landmark_results.face_landmarks.landmark:
                face_lm = np.array([lm.x, lm.y, lm.z])
                frame_face_lm.append(face_lm)

        # Get landmarks for left hand
        if landmark_results.left_hand_landmarks is None:
            frame_left_hand_lm = np.zeros((21,3))
        else:
            for lm in landmark_results.left_hand_landmarks.landmark:
                left_hand_lm = np.array([lm.x, lm.y, lm.z])
                frame_left_hand_lm.append(left_hand_lm)

        # Get landmarks for right hand
        if landmark_results.right_hand_landmarks is None:
            frame_right_hand_lm = np.zeros((21,3))
        else:
            for lm in landmark_results.right_hand_landmarks.landmark:
                right_hand_lm = np.array([lm.x, lm.y, lm.z])
                frame_right_hand_lm.append(right_hand_lm)

        #Convert all arrays to numpy arrays
        frame_pose_lm_np = np.asarray(frame_pose_lm)
        frame_face_lm_np = np.asarray(frame_face_lm)
        frame_left_hand_lm_np = np.asarray(frame_left_hand_lm)
        frame_right_hand_lm_np = np.asarray(frame_right_hand_lm)

        # Print shapes for debugging
        #print(frame_pose_lm_np.shape)
        #print(frame_face_lm_np.shape)
        #print(frame_left_hand_lm_np.shape)
        #print(frame_right_hand_lm_np.shape)

        # Flatten all arrays
        frame_pose_lm_np_flat = frame_pose_lm_np.flatten()
        frame_face_lm_np_flat = frame_face_lm_np.flatten()
        frame_left_hand_lm_np_flat = frame_left_hand_lm_np.flatten()
        frame_right_hand_lm_np_flat = frame_right_hand_lm_np.flatten()

        #Combine all landmarks to one big array.
        return np.concatenate([frame_pose_lm_np_flat, frame_face_lm_np_flat, frame_left_hand_lm_np_flat, frame_right_hand_lm_np_flat])
    
    def words_frames_to_points(self, start_word=0, end_word=1):
        count = 0
        for word_id in range(start_word, end_word):
            for video_id in range(self.videos_per_word):
                frames_base_dir = os.path.join('video_data',f'{word_id}', f'{video_id}', 'frames')
                #print(frames_base_dir)

                # Create directory to store landmark points
                points_base_dir = os.path.join('video_data',f'{word_id}', f'{video_id}', 'landmarks')
                if os.path.exists(points_base_dir) is False:
                    os.makedirs(points_base_dir)
                    #print(f'creating {points_base_dir}')
                else:
                    pass
                    #print(f'{points_base_dir} already exists')
                
                # Extract landmark points of each frame
                for num_frame in range(self.frames_per_video):
                    frame_file_path = os.path.join(frames_base_dir, f'{word_id}_{video_id}_{num_frame}.png')
                    #print(f'Extracting landmark points for {frame_file_path}')

                    numpy_save_file_path = os.path.join(points_base_dir, f'{word_id}_{video_id}_{num_frame}.npy')
                    #print(f'Saving numpy array to {numpy_save_file_path}')

                    # Extract landmark points for this frame
                    img, lm_res = self.open_image_detect_holistic(frame_file_path)

                    # Retrieve the landmark points in one big array
                    lm_frame = self.retrieve_landmarks(lm_res)
                    
                    # Incase something goes wrong and the combined landmark array is not the correct shape,
                    # show an error message and don't save the array
                    if lm_frame.shape != (1662,):
                        print(f'{word_id}_{video_id}_{num_frame} has missing landmark results')
                    else:
                        pass
                        np.save(numpy_save_file_path, lm_frame)
                        count += 1
        return count
    
    def pre_process_words(self, start_word=0, end_word=1):
        num_frames_saved = self.word_videos_to_frames(start_word, end_word)
        print(f' Saved {num_frames_saved} frames for {end_word - start_word} words')

        num_landmarks_saved = self.words_frames_to_points(start_word, end_word)
        print(f'Saved {num_landmarks_saved} landmark arrays for {end_word - start_word} words')