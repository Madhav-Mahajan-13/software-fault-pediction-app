import tkinter as tk
import tensorflow as tf
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# TensorFlow and Keras imports


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
import joblib

class DataAnalysisApp:
    sampling_methods = [
        "No sampling",
        "Over sampling (SMOTE)",
        "Under sampling",
        "SMOTE",
        "ADASYN",
        "SMOTE + ADASYN",
        "Balanced Bagging Classifier",
        "Random Under-Sampling",
        "SMOTE + Tomek Links",
        "SMOTE + ENN",
        "Cluster Centroids"
    ]

    model_options = [
        "CNN + LSTM",
        "Random Forest + Gradient Boosting",
        "K-Means Clustering + Principal Component Analysis (PCA)",
        "Autoencoder + Support Vector Machine (SVM)",
        "Decision Tree + Logistic Regression",
        "Word2Vec + Recurrent Neural Network (RNN)",
        "Naive Bayes + k-Nearest Neighbors (k-NN)",
        "Reinforcement Learning + Deep Q-Learning",
        "Gradient Boosting + Neural Network",
        "Genetic Algorithm + Neural Network"
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis App")

        # Variables
        self.training_files = []
        self.common_features = []
        self.sampling_needed = tk.BooleanVar()
        self.sampling_option = tk.StringVar()
        self.sampling_option.set(self.sampling_methods[0])  # Set default value
        self.selected_model = tk.StringVar()
        self.selected_model.set(self.model_options[0])  # Set default value

        # Styling
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TFrame", padding="10")
        style.configure("TEntry", font=("Helvetica", 10))
        style.configure("TCheckbutton", font=("Helvetica", 10))
        style.configure("TOptionMenu", font=("Helvetica", 10))

        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # UI Elements
        label_train_files = ttk.Label(main_frame, text="Upload CSV Files for Training:")
        button_browse_train = ttk.Button(main_frame, text="Browse", command=self.browse_train_files)

        check_sampling = ttk.Checkbutton(main_frame, text="Perform Sampling", variable=self.sampling_needed)
        option_sampling = ttk.OptionMenu(main_frame, self.sampling_option, *self.sampling_methods)

        label_model = ttk.Label(main_frame, text="Select Model:")
        option_model = ttk.OptionMenu(main_frame, self.selected_model, *self.model_options)

        button_train = ttk.Button(main_frame, text="Train Model", command=self.train_model)

        # Layout
        label_train_files.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        button_browse_train.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        check_sampling.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        option_sampling.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        label_model.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        option_model.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
        button_train.grid(row=3, column=0, columnspan=2, padx=10, pady=20)
        button_predict = ttk.Button(main_frame, text="Predict on New Data", command=self.predict_faults)
        button_predict.grid(row=4, column=0, columnspan=2, padx=10, pady=20)

    def browse_train_files(self):
        self.training_files = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
        if self.training_files:
            messagebox.showinfo("Files Selected", f"Selected training files: {self.training_files}")
            self.process_data()

    def process_data(self):
        if not self.training_files:
            messagebox.showerror("Error", "Please select CSV files for training.")
            return

        try:
            # Load training data
            dfs_train = [pd.read_csv(file) for file in self.training_files]

            # Ensure all dataframes have the 'faults' column
            if not all('faults' in df.columns for df in dfs_train):
                messagebox.showerror("Error", "All datasets must contain a 'faults' column.")
                return

            # Merge training data
            merged_train_df = pd.concat(dfs_train, ignore_index=True)

            # Handle missing and null values
            merged_train_df = self.handle_missing_values(merged_train_df)

            # Separate features and target
            self.X_train = merged_train_df.drop('faults', axis=1)
            self.y_train = merged_train_df['faults']

            self.feature_names = list(self.X_train.columns)

            messagebox.showinfo("Data Processed", f"Data has been processed.\n"
                                                  f"Number of samples: {len(self.X_train)}\n"
                                                  f"Number of features: {len(self.feature_names)}")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred while processing data: {str(e)}")

    def handle_missing_values(self, df):
        # Check for missing values
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()

        if total_missing > 0:
            messagebox.showinfo("Missing Values", f"Total missing values: {total_missing}\n\nMissing values by column:\n{missing_values}")

            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            imputer_numeric = SimpleImputer(strategy='mean')
            df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])

            # Handle categorical columns
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])

            messagebox.showinfo("Missing Values Handled", "Missing values have been imputed.")
        else:
            messagebox.showinfo("Data Quality", "No missing values found in the dataset.")

        return df

    def apply_sampling(self, X, y):
        sampling_option = self.sampling_option.get()
        if sampling_option == "Over sampling (SMOTE)":
            sampler = SMOTE(random_state=42)
        elif sampling_option == "Under sampling":
            sampler = RandomUnderSampler(random_state=42)
        elif sampling_option == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif sampling_option == "ADASYN":
            sampler = ADASYN(random_state=42)
        elif sampling_option == "SMOTE + ADASYN":
            sampler = SMOTEENN(random_state=42)
        elif sampling_option == "Balanced Bagging Classifier":
            # This is handled differently as it's a classifier, not just a sampler
            model = BalancedRandomForestClassifier(random_state=42)
            model.fit(X, y)
            return X, y  # Return original data as this is a complete model
        elif sampling_option == "Random Under-Sampling":
            sampler = RandomUnderSampler(random_state=42)
        elif sampling_option == "SMOTE + Tomek Links":
            sampler = SMOTETomek(random_state=42)
        elif sampling_option == "SMOTE + ENN":
            sampler = SMOTEENN(random_state=42)
        elif sampling_option == "Cluster Centroids":
            sampler = ClusterCentroids(random_state=42)
        else:  # "No sampling"
            return X, y

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def train_model(self):
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            messagebox.showerror("Error", "Please process the data before training the model.")
            return

        selected_model = self.selected_model.get()
        messagebox.showinfo("Model Training", f"Training model: {selected_model}")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if selected_model == "CNN + LSTM":
            # Reshape input for CNN-LSTM
            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
                LSTM(64),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

            y_pred = model.predict(X_test_reshaped)
            y_pred_classes = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Random Forest + Gradient Boosting":
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)

            rf_pred = rf_model.predict(X_test_scaled)
            gb_pred = gb_model.predict(X_test_scaled)

            ensemble_pred = np.mean([rf_pred, gb_pred], axis=0)
            y_pred_classes = (ensemble_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "K-Means Clustering + Principal Component Analysis (PCA)":
            pca = PCA(n_components=0.95)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(X_train_pca)

            y_pred_classes = kmeans.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Autoencoder + Support Vector Machine (SVM)":
            # Simple autoencoder
            input_dim = X_train_scaled.shape[1]
            encoding_dim = 32

            autoencoder = Sequential([
                Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
                Dense(input_dim, activation='sigmoid')
            ])
            autoencoder.compile(optimizer='adam', loss='mse')

            autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, validation_split=0.2)

            encoder = Sequential(autoencoder.layers[:1])
            X_train_encoded = encoder.predict(X_train_scaled)
            X_test_encoded = encoder.predict(X_test_scaled)

            svm_model = SVC(kernel='rbf', random_state=42)
            svm_model.fit(X_train_encoded, y_train)

            y_pred_classes = svm_model.predict(X_test_encoded)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Decision Tree + Logistic Regression":
            dt_model = DecisionTreeClassifier(random_state=42)
            lr_model = LogisticRegression(random_state=42)

            dt_model.fit(X_train_scaled, y_train)
            lr_model.fit(X_train_scaled, y_train)

            dt_pred = dt_model.predict(X_test_scaled)
            lr_pred = lr_model.predict(X_test_scaled)

            ensemble_pred = np.mean([dt_pred, lr_pred], axis=0)
            y_pred_classes = (ensemble_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Word2Vec + Recurrent Neural Network (RNN)":
            messagebox.showinfo("Model Info",
                                "This model is typically used for text data. Using a simple RNN for demonstration.")

            model = Sequential([
                LSTM(64, input_shape=(X_train_scaled.shape[1], 1)),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

            model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

            y_pred = model.predict(X_test_reshaped)
            y_pred_classes = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Naive Bayes + k-Nearest Neighbors (k-NN)":
            nb_model = GaussianNB()
            knn_model = KNeighborsClassifier(n_neighbors=5)

            nb_model.fit(X_train_scaled, y_train)
            knn_model.fit(X_train_scaled, y_train)

            nb_pred = nb_model.predict(X_test_scaled)
            knn_pred = knn_model.predict(X_test_scaled)

            ensemble_pred = np.mean([nb_pred, knn_pred], axis=0)
            y_pred_classes = (ensemble_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Reinforcement Learning + Deep Q-Learning":
            messagebox.showinfo("Model Info",
                                "Reinforcement Learning typically requires a specific environment. Using a simple neural network for demonstration.")

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

            y_pred = model.predict(X_test_scaled)
            y_pred_classes = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Gradient Boosting + Neural Network":
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)

            gb_model.fit(X_train_scaled, y_train)
            nn_model.fit(X_train_scaled, y_train)

            gb_pred = gb_model.predict(X_test_scaled)
            nn_pred = nn_model.predict(X_test_scaled)

            ensemble_pred = np.mean([gb_pred, nn_pred], axis=0)
            y_pred_classes = (ensemble_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        elif selected_model == "Genetic Algorithm + Neural Network":
            messagebox.showinfo("Model Info",
                                "Genetic Algorithm typically requires specific implementation. Using a simple neural network for demonstration.")

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

            y_pred = model.predict(X_test_scaled)
            y_pred_classes = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)

        accuracy = accuracy_score(y_test, y_pred_classes)
        classification_rep = classification_report(y_test, y_pred_classes)
        self.trained_model = model
        self.scaler = scaler
        # Display results
        messagebox.showinfo("Training Complete",
                            f"Model {selected_model} has been trained successfully.\n"
                            f"Accuracy: {accuracy:.2f}\n\n"
                            f"Classification Report:\n{classification_rep}")

    def display_feature_importance(self, model):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            feature_importance_text = "Feature Importances:\n"
            for f, idx in enumerate(indices):
                feature_importance_text += f"{self.feature_names[idx]}: {importances[idx]:.4f}\n"

            messagebox.showinfo("Feature Importances", feature_importance_text)
        else:
            messagebox.showinfo("Feature Importances", "This model doesn't provide feature importances.")

    def save_model(self, model, scaler):
        file_path = filedialog.asksaveasfilename(defaultextension=".joblib")
        if file_path:
            joblib.dump({'model': model, 'scaler': scaler}, file_path)
            messagebox.showinfo("Model Saved", f"Model and scaler saved to {file_path}")

    def load_new_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return None

        try:
            new_df = pd.read_csv(file_path)

            # Check if all common features are present in the new dataset
            missing_features = set(self.feature_names) - set(new_df.columns)
            if missing_features:
                messagebox.showerror("Error", f"The new dataset is missing these features: {missing_features}")
                return None

            # Select only the common features
            new_df = new_df[self.feature_names]

            # Handle missing values
            new_df = self.handle_missing_values(new_df)

            return new_df

        except Exception as e:
            messagebox.showerror("Error", f"Error loading new dataset: {str(e)}")
            return None

    def predict_faults(self):
        if not hasattr(self, 'trained_model') or not hasattr(self, 'scaler'):
            messagebox.showerror("Error", "Please train a model first.")
            return

        new_df = self.load_new_dataset()
        if new_df is None:
            return

        # Scale the features
        X_new_scaled = self.scaler.transform(new_df)

        # Predict faults
        y_pred = self.trained_model.predict(X_new_scaled)

        # Add predictions to the dataframe
        new_df['faults'] = y_pred

        # Calculate number of faults
        num_faults = np.sum(y_pred)

        # Save the results
        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            new_df.to_csv(save_path, index=False)
            messagebox.showinfo("Prediction Complete",
                                f"Predictions saved to {save_path}\n"
                                f"Number of faults predicted: {num_faults}")
        else:
            messagebox.showinfo("Prediction Complete",
                                f"Number of faults predicted: {num_faults}")




if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()






