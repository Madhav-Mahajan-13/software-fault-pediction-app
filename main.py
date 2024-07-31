import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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

    sampling_methods = [
        "Over sampling",
        "Under sampling",
        "SMOTE",
        "ADASYN",
        "SMOTE + ADASYN"
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

        # Styling
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TCheckbutton", font=("Helvetica", 10))
        style.configure("TOptionMenu", font=("Helvetica", 10))
        style.configure("TFrame", padding="10")
        style.configure("TEntry", font=("Helvetica", 10))

        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # UI Elements
        label_train_files = ttk.Label(main_frame, text="Upload CSV Files for Training:")
        button_browse_train = ttk.Button(main_frame, text="Browse", command=self.browse_train_files)

        label_test_file = ttk.Label(main_frame, text="Select CSV File for Testing:")
        button_browse_test = ttk.Button(main_frame, text="Browse", command=self.browse_test_file)

        check_sampling = ttk.Checkbutton(main_frame, text="Perform Sampling if Imbalanced", variable=self.sampling_needed)
        option_sampling = ttk.OptionMenu(main_frame, self.sampling_option, *self.sampling_methods)

        label_model = ttk.Label(main_frame, text="Select Model:")
        option_model = ttk.OptionMenu(main_frame, self.selected_model, *self.model_options)

        button_train = ttk.Button(main_frame, text="Train Model", command=self.train_model)
        button_compare = ttk.Button(main_frame, text="Compare Models", command=self.compare_models)

        # Layout
        label_train_files.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        button_browse_train.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        label_test_file.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        button_browse_test.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        check_sampling.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        option_sampling.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

        label_model.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        option_model.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)

        button_train.grid(row=4, column=0, padx=10, pady=20, sticky=tk.W)
        button_compare.grid(row=4, column=1, padx=10, pady=20, sticky=tk.W)

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

            if not common_columns:
                messagebox.showerror("Error", "No common columns found in the training datasets.")
                return

            common_columns = list(common_columns)
            # Merge training data
            merged_train_df = pd.concat([df[common_columns] for df in dfs_train])

            # Handle missing values in training data
            merged_train_df.fillna(merged_train_df.mean(), inplace=True)

            # Select features and target
            feature_columns = merged_train_df.columns[:-1]  # Assuming the last column is the target
            target_column = merged_train_df.columns[-1]

            X_train = merged_train_df[feature_columns]
            y_train = merged_train_df[target_column]

            # Load test data
            df_test = pd.read_csv(self.test_file)
            df_test = df_test[common_columns]  # Ensure test data has the same columns

            # Handle missing values in test data
            df_test.fillna(df_test.mean(), inplace=True)

            X_test = df_test[feature_columns]

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

                messagebox.showinfo("Model Evaluation",
                                    f"Random Forest Accuracy: {accuracy_score(y_train, rf_pred):.2f}\nGradient Boosting Accuracy: {accuracy_score(y_train, gb_pred):.2f}")

            # Implement other models similarly

        except ValueError as ve:
            if "Expected n_neighbors" in str(ve):
                messagebox.showerror("Error",
                                     f"Error occurred: {str(ve)}. Reduce the number of neighbors or increase the dataset size.")
            else:
                messagebox.showerror("Error", f"Error occurred: {str(ve)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred: {str(e)}")

    def compare_models(self):
        # Implement comparison of model results
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
