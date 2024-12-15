import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Function to load class names from labels.txt
def load_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.read().splitlines()
        #print(lines)
        word_dict = {}
        for i in range(len(lines)):
            id_word = lines[i].split(',')
            word_dict[id_word[0]] = id_word[1]
    return word_dict

def retrieve_landmarks(landmark_results):
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

def detect_holistic(holistic_model, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the color to RGB to pass into the holistic model
    img.flags.writeable = False # Avoid any bugs by not allowing the image to be modified
    landmarks = holistic_model.process(img) # Extract landmarks from the image
    img.flags.writeable = True  # Make the image writeable again
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Conver the color back to the original BGR format

    return landmarks

def main():
    # Load the trained model
    model = tf.keras.models.load_model('asl_11words_model_4L2D_93acc.keras')

    mp_holistic = mp.solutions.holistic

    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    labels_dict = load_labels('words.txt')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Getting frame width and height for video saving
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # codec and VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("live_asl_demo.mp4", fourcc, 15, (frame_width, frame_height))
    frames_sequence = []
    full_sentence = ''
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break


        # Predict the class
        landmarks = detect_holistic(holistic_model, frame)
        landmark_array = retrieve_landmarks(landmark_results=landmarks)
        frames_sequence.append(landmark_array)

        if len(frames_sequence) == 70:
            cv2.waitKey(1000)
            print(f'Predicting....')
            expanded_sequence = np.expand_dims(frames_sequence, axis=0)
            predictions_prob = model.predict(expanded_sequence)[0]
            sorted_probs = np.argsort(predictions_prob)
            top_three_ind = sorted_probs[-3:]
            top_three_ind = np.flip(top_three_ind)
            top_three_vals = predictions_prob[top_three_ind]
            print(f'top 3: {top_three_ind} with confidence {top_three_vals}')
            if top_three_vals[0] > 0.8:
                top_word = labels_dict.get(str(top_three_ind[0]), "?")
                #print(predictions_prob)
                full_sentence = full_sentence + str(top_word) + ' '
                print(f'predicted word: {top_word}')
                cv2.putText(frame, f'Predicted {top_word}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('ASL_Demo', frame)
                cv2.waitKey(1000)
            else:
                cv2.putText(frame, f'Low confidence', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('ASL_Demo', frame)
                cv2.waitKey(1000)
                print('too low confidence.')
            print(f'sentence so far: {full_sentence}')
            frames_sequence.clear()
        else:
            # Display the resulting frame with prediction
            cv2.putText(frame, f'Collecting Frames', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('ASL_Demo', frame)

        # Write frame to the output video file
        out_video.write(frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and video writer
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()