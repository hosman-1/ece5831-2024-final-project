import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

class Training_ASL():
    num_videos = 100
    videos_base_dir = 'video_data'

    def __init__(self, start_frames=5, end_frames=75, num_words=5):
        self.num_words = num_words
        self.start_frames = start_frames
        self.end_frames = end_frames

        self._load_data()

    def _load_data(self, videos_per_word=100, start_frame=5, end_frame=75):
        data = []
        labels = []
        for words in range(self.num_words):
            for videos in range(videos_per_word):
                video_array_list = []
                for frames in range(start_frame, end_frame):
                    numpy_array_file_path = os.path.join(self.videos_base_dir,f'{words}',f'{videos}', 'landmarks', f'{words}_{videos}_{frames}.npy')
                    temp_array = np.load(numpy_array_file_path)
                    #print(f'Loading {numpy_array_file_path}')
                    video_array_list.append(temp_array)
                data.append(video_array_list)
                labels.append(words)
        self.data_np = np.array(data)
        print(f'Shape of data: {self.data_np.shape}')

        self.labels_np = np.array(labels)
        print(f'Shape of labels: {self.labels_np.shape}')
    
    def random_split_prepare_data(self, train_split=0.7, val_split=0.15, test_split=0.15, rng_seed=113):
        rng = np.random.RandomState(rng_seed)
        data_indices = np.arange(len(self.data_np))
        #print(data_indices)
        rng.shuffle(data_indices)

        train_split_max = int(train_split * len(self.data_np))

        val_split_max = int(train_split_max + (len(self.data_np) * val_split))

        test_split_max = int(val_split_max + (len(self.data_np) * test_split))

        train_indices = data_indices[0:train_split_max]
        val_indices = data_indices[train_split_max:val_split_max]
        test_indices = data_indices[val_split_max:test_split_max]

        self.x_train = self.data_np[train_indices]
        self.y_train = self.labels_np[train_indices]

        self.x_val = self.data_np[val_indices]
        self.y_val = self.labels_np[val_indices]

        self.x_test = self.data_np[test_indices]
        self.y_test = self.labels_np[test_indices]

        self.y_train = to_categorical(self.y_train)
        self.y_val = to_categorical(self.y_val)
        self.y_test = to_categorical(self.y_test)

    def build_model(self):
        self.model = Sequential([
            LSTM(32, return_sequences=True, activation='relu', input_shape=(70,1662)),
            LSTM(64, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False),
            Dense(128, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.num_words, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=200, batch_size=32, validation_data=(self.x_val, self.y_val))

    def plot_loss(self):
        history_dict = self.history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, "r--", label="Training loss")
        plt.plot(epochs, val_loss_values, "b", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    def plot_accuracy(self):
        history_dict = self.history.history
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, "r--", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="Validation acc")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    
    def evaluate_test(self):
        results = self.model.evaluate(self.x_test, self.y_test)
        print(f'Loss: {results[0]}\nAccuracy: {results[1]}')

    def save_model(self, save_path='asl_words_model_2L2D.keras'):
        self.model.save(filepath=save_path)

    def predict(self, video_array, label):
        prediction = np.argmax(self.model.predict(video_array.reshape((1,70,1662))))
        if prediction == label:
            print(f'Successfully predicted {prediction}, which matches the label {label}')
        else:
            print(f'Incorrectly predicted {prediction}, but the real label is {label}')