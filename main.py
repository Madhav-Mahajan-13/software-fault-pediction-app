import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class DataAnalysisApp:
    model_options = [
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
        self.test_file = None
        self.sampling_needed = tk.BooleanVar()
        self.sampling_option = tk.StringVar()
        self.selected_model = tk.StringVar()

        # UI Elements
        self.label_train_files = tk.Label(self.root, text="Upload CSV Files for Training:")
        self.button_browse_train = tk.Button(self.root, text="Browse", command=self.browse_train_files)
        self.label_test_file = tk.Label(self.root, text="Select CSV File for Testing:")
        self.button_browse_test = tk.Button(self.root, text="Browse", command=self.browse_test_file)
        self.check_sampling = tk.Checkbutton(self.root, text="Perform Sampling if Imbalanced", variable=self.sampling_needed)
        self.option_sampling = tk.OptionMenu(self.root, self.sampling_option, *["Over sampling", "Under sampling", "SMOTE", "ADASYN", "SMOTE + ADASYN"])
        self.label_model = tk.Label(self.root, text="Select Model:")
        self.option_model = tk.OptionMenu(self.root, self.selected_model, *self.model_options)
        self.button_train = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.button_compare = tk.Button(self.root, text="Compare Models", command=self.compare_models)

        # Layout
        self.label_train_files.grid(row=0, column=0, padx=10, pady=10)
        self.button_browse_train.grid(row=0, column=1, padx=10, pady=10)
        self.label_test_file.grid(row=1, column=0, padx=10, pady=10)
        self.button_browse_test.grid(row=1, column=1, padx=10, pady=10)
        self.check_sampling.grid(row=2, columnspan=2, padx=10, pady=10, sticky=tk.W)
        self.option_sampling.grid(row=3, columnspan=2, padx=10, pady=10, sticky=tk.W)
        self.label_model.grid(row=4, column=0, padx=10, pady=10)
        self.option_model.grid(row=4, column=1, padx=10, pady=10)
        self.button_train.grid(row=5, columnspan=2, padx=10, pady=10)
        self.button_compare.grid(row=6, columnspan=2, padx=10, pady=10)

    def browse_train_files(self):
        self.training_files = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
        if self.training_files:
            messagebox.showinfo("Files Selected", f"Selected training files: {self.training_files}")

    def browse_test_file(self):
        self.test_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.test_file:
            messagebox.showinfo("File Selected", f"Selected test file: {self.test_file}")

    def train_model(self):
        if not self.training_files:
            messagebox.showerror("Error", "Please select CSV files for training.")
            return

        if not self.test_file:
            messagebox.showerror("Error", "Please select a CSV file for testing.")
            return

        try:
            # Load training data
            dfs_train = [pd.read_csv(file) for file in self.training_files]
            common_columns = set(dfs_train[0].columns)

            for df in dfs_train[1:]:
                common_columns.intersection_update(df.columns)

            # Select common features for training
            feature_name = list(common_columns)[0] if common_columns else None

            X_train = pd.concat([df[feature_name] for df in dfs_train], axis=1)
            y_train = dfs_train[0][feature_name]

            # Handle missing values in training data
            X_train = X_train.fillna(X_train.mean())  # Replace NaN with mean (you can use other strategies as well)

            # Load test data
            df_test = pd.read_csv(self.test_file)
            X_test = df_test[feature_name]

            # Handle missing values in test data
            X_test = X_test.fillna(X_test.mean())  # Replace NaN with mean (you can use other strategies as well)

            # Perform sampling if needed
            if self.sampling_needed.get():
                sampling_option = self.sampling_option.get()
                if sampling_option == "Over sampling":
                    sampler = SMOTE(random_state=42)
                elif sampling_option == "Under sampling":
                    sampler = RandomUnderSampler(random_state=42)
                elif sampling_option == "SMOTE":
                    sampler = SMOTE(random_state=42)
                elif sampling_option == "ADASYN":
                    sampler = ADASYN(random_state=42)
                elif sampling_option == "SMOTE + ADASYN":
                    sampler = SMOTEENN(random_state=42)

                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                X_train = X_resampled
                y_train = y_resampled

            # Train the selected model
            selected_model = self.selected_model.get()

            if selected_model == "Random Forest + Gradient Boosting":
                rf_model = RandomForestClassifier(random_state=42)
                gb_model = GradientBoostingClassifier(random_state=42)

                rf_model.fit(X_train, y_train)
                gb_model.fit(X_train, y_train)

                # Evaluate models
                rf_pred = rf_model.predict(X_test)
                gb_pred = gb_model.predict(X_test)

                messagebox.showinfo("Model Evaluation", f"Random Forest Accuracy: {accuracy_score(df_test[feature_name], rf_pred):.2f}\nGradient Boosting Accuracy: {accuracy_score(df_test[feature_name], gb_pred):.2f}")

            # Implement other models similarly

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred: {str(e)}")

    def compare_models(self):
        # Implement comparison of model results
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
