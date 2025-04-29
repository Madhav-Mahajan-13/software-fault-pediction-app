import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os


class DataAnalysisApp:
    model_options = [
        "Random Forest + Gradient Boosting",
        "K-Means Clustering + Principal Component Analysis (PCA)",
        "Support Vector Machine (SVM)",
        "Decision Tree + Logistic Regression",
        "Naive Bayes + k-Nearest Neighbors (k-NN)",
        "Neural Network"
    ]

    sampling_methods = [
        "Over sampling",
        "Under sampling",
        "SMOTE",
        "ADASYN",
        "SMOTE + ENN"
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Software Fault Prediction System")
        self.root.geometry("600x400")

        # Variables
        self.training_files = []
        self.test_file = None
        self.sampling_needed = tk.BooleanVar()
        self.sampling_option = tk.StringVar(value=self.sampling_methods[0])
        self.selected_model = tk.StringVar(value=self.model_options[0])
        self.results = {}

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

        check_sampling = ttk.Checkbutton(main_frame, text="Perform Sampling if Imbalanced",
                                         variable=self.sampling_needed)

        label_sampling = ttk.Label(main_frame, text="Sampling Method:")
        option_sampling = ttk.OptionMenu(main_frame, self.sampling_option, *self.sampling_methods)

        label_model = ttk.Label(main_frame, text="Select Model:")
        option_model = ttk.OptionMenu(main_frame, self.selected_model, *self.model_options)

        # Add buttons
        button_train = ttk.Button(main_frame, text="Train Model", command=self.train_model)
        button_compare = ttk.Button(main_frame, text="Compare Models", command=self.compare_models)
        button_view_results = ttk.Button(main_frame, text="View Results", command=self.view_prediction_results)

        # Status message
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Helvetica", 10, "italic"))

        # Layout
        label_train_files.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        button_browse_train.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        label_test_file.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        button_browse_test.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        check_sampling.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)

        label_sampling.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        option_sampling.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)

        label_model.grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
        option_model.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)

        button_train.grid(row=5, column=0, padx=10, pady=20, sticky=tk.W)
        button_compare.grid(row=5, column=1, padx=10, pady=20, sticky=tk.W)
        button_view_results.grid(row=5, column=2, padx=10, pady=20, sticky=tk.W)

        status_label.grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky=tk.W)

    def browse_train_files(self):
        self.training_files = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
        if self.training_files:
            filenames = [file.split('/')[-1].split('\\')[-1] for file in self.training_files]
            self.status_var.set(f"Selected {len(self.training_files)} training files")
            messagebox.showinfo("Files Selected", f"Selected training files: {', '.join(filenames)}")

    def browse_test_file(self):
        self.test_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.test_file:
            filename = self.test_file.split('/')[-1].split('\\')[-1]
            self.status_var.set(f"Test file selected: {filename}")
            messagebox.showinfo("File Selected", f"Selected test file: {filename}")

    def prepare_data(self):
        """Prepare data from potentially heterogeneous datasets"""
        try:
            # Load training data
            dfs_train = []
            for file in self.training_files:
                df = pd.read_csv(file)
                # Ensure label column exists and is properly named
                if 'label' not in df.columns and 'Label' in df.columns:
                    df.rename(columns={'Label': 'label'}, inplace=True)
                # Handle upper/lowercase column name variations in NASA datasets
                if 'label' not in df.columns and any(col for col in df.columns if col.lower() == 'label'):
                    for col in df.columns:
                        if col.lower() == 'label':
                            df.rename(columns={col: 'label'}, inplace=True)
                dfs_train.append(df)

            self.status_var.set(f"Processing {len(dfs_train)} datasets...")

            # Find common columns if needed, otherwise just use all columns from each dataset
            use_all_columns = True  # Flag to determine if we'll use all columns

            # Check if datasets have compatible structure (similar columns)
            first_df_cols = set(dfs_train[0].columns)
            for df in dfs_train[1:]:
                common = first_df_cols.intersection(set(df.columns))
                # If datasets have very few common columns, we'll need to use feature vectors separately
                if len(common) < 5:  # Arbitrary threshold
                    use_all_columns = False
                    break

            if use_all_columns:
                # Process datasets with common structure - find common columns
                common_columns = set(dfs_train[0].columns)
                for df in dfs_train[1:]:
                    common_columns.intersection_update(df.columns)

                # Always ensure 'label' is in the common columns
                if 'label' not in common_columns:
                    raise ValueError("No 'label' column found in common columns across datasets")

                common_columns = list(common_columns)
                # Merge training data on common columns
                merged_train_df = pd.concat([df[common_columns] for df in dfs_train])
            else:
                # Process heterogeneous datasets - concatenate and handle missing values properly
                # First, make sure each dataset has a 'label' column
                for df in dfs_train:
                    if 'label' not in df.columns:
                        raise ValueError(f"No 'label' column found in one of the datasets")

                # For each dataset, keep only the 'label' and fill the rest with NaNs
                processed_dfs = []
                all_columns = set()
                for df in dfs_train:
                    all_columns.update(df.columns)

                all_columns = list(all_columns)

                for df in dfs_train:
                    # Create a new dataframe with all columns, filled with NaNs
                    new_df = pd.DataFrame(columns=all_columns)
                    # Add the data from this dataset
                    for col in df.columns:
                        new_df[col] = df[col]
                    processed_dfs.append(new_df)

                # Concatenate all datasets
                merged_train_df = pd.concat(processed_dfs, ignore_index=True)
                common_columns = all_columns

            # Handle missing values in merged training data
            for col in merged_train_df.columns:
                if col != 'label':  # Skip the label column
                    if merged_train_df[col].dtype.kind in 'iuf':  # if column is numeric
                        merged_train_df[col] = merged_train_df[col].fillna(merged_train_df[col].mean())
                    else:
                        merged_train_df[col] = merged_train_df[col].fillna(
                            merged_train_df[col].mode()[0] if not merged_train_df[col].mode().empty else "MISSING")

            # Select features and target
            feature_columns = [col for col in merged_train_df.columns if col != 'label']
            target_column = 'label'

            # Create explicit copies to avoid SettingWithCopyWarning
            X_train = merged_train_df[feature_columns].copy()
            y_train = merged_train_df[target_column].copy()

            # Convert y_train to discrete class labels if it's continuous
            if y_train.dtype.kind == 'f':
                # Assume binary classification threshold at 0.5
                y_train = (y_train > 0.5).astype(int)

            # Load test data
            df_test = pd.read_csv(self.test_file)

            # Ensure label column is properly named in test data
            if 'label' not in df_test.columns and 'Label' in df_test.columns:
                df_test.rename(columns={'Label': 'label'}, inplace=True)

            # Handle upper/lowercase column name variations in NASA datasets
            if 'label' not in df_test.columns and any(col for col in df_test.columns if col.lower() == 'label'):
                for col in df_test.columns:
                    if col.lower() == 'label':
                        df_test.rename(columns={col: 'label'}, inplace=True)

            # Add missing columns to test data (that exist in training but not in test)
            missing_cols = set(feature_columns) - set(df_test.columns)
            for col in missing_cols:
                df_test[col] = 0  # Fill with zeros or other default value

            # Create a new DataFrame explicitly to avoid SettingWithCopyWarning
            X_test = pd.DataFrame(df_test[feature_columns].copy())

            # Handle missing values in test data
            for col in X_test.columns:
                if X_test[col].dtype.kind in 'iuf':
                    # Use .loc to avoid SettingWithCopyWarning
                    X_test.loc[:, col] = X_test[col].fillna(X_test[col].mean() if not X_test[col].isna().all() else 0)
                else:
                    X_test.loc[:, col] = X_test[col].fillna(
                        X_test[col].mode()[0] if not X_test[col].mode().empty else "MISSING")

            # Scale features if needed (optional but recommended for heterogeneous datasets)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            # Perform sampling if needed
            if self.sampling_needed.get():
                sampling_option = self.sampling_option.get()
                try:
                    if sampling_option == "Over sampling":
                        sampler = SMOTE(random_state=42)
                    elif sampling_option == "Under sampling":
                        sampler = RandomUnderSampler(random_state=42)
                    elif sampling_option == "SMOTE":
                        sampler = SMOTE(random_state=42)
                    elif sampling_option == "ADASYN":
                        sampler = ADASYN(random_state=42)
                    elif sampling_option == "SMOTE + ENN":
                        sampler = SMOTEENN(random_state=42)

                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                    X_train = X_resampled
                    y_train = y_resampled
                    self.status_var.set(f"Applied {sampling_option}")
                except ValueError as e:
                    if "Expected n_neighbors" in str(e):
                        # If not enough samples in minority class, try reducing n_neighbors
                        if sampling_option in ["SMOTE", "ADASYN", "SMOTE + ENN"]:
                            min_samples = y_train.value_counts().min()
                            if min_samples < 5:
                                k_neighbors = max(1, min_samples - 1)
                                if sampling_option == "SMOTE":
                                    sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
                                elif sampling_option == "ADASYN":
                                    sampler = ADASYN(random_state=42, n_neighbors=k_neighbors)
                                elif sampling_option == "SMOTE + ENN":
                                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                                    sampler = SMOTEENN(random_state=42, smote=smote)

                                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                                X_train = X_resampled
                                y_train = y_resampled
                                self.status_var.set(f"Applied {sampling_option} with reduced neighbors")
                            else:
                                raise ValueError(
                                    f"Error in sampling: {str(e)}. Try a different sampling method for this dataset.")
                    else:
                        raise ValueError(f"Error in sampling: {str(e)}")

            return X_train, y_train, X_test, feature_columns, target_column, df_test

        except Exception as e:
            raise Exception(f"Error in data preparation: {str(e)}")

    def train_model(self):
        if not self.training_files:
            messagebox.showerror("Error", "Please select CSV files for training.")
            return

        if not self.test_file:
            messagebox.showerror("Error", "Please select a CSV file for testing.")
            return

        try:
            self.status_var.set("Preparing data...")
            X_train, y_train, X_test, feature_columns, target_column, original_test_df = self.prepare_data()

            # Split training data for validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            # Train the selected model
            selected_model = self.selected_model.get()
            self.status_var.set(f"Training {selected_model}...")

            if selected_model == "Random Forest + Gradient Boosting":
                # Random Forest
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train_split, y_train_split)
                rf_val_pred = rf_model.predict(X_val)
                rf_accuracy = accuracy_score(y_val, rf_val_pred)
                rf_precision = precision_score(y_val, rf_val_pred, average='weighted', zero_division=0)
                rf_recall = recall_score(y_val, rf_val_pred, average='weighted', zero_division=0)
                rf_f1 = f1_score(y_val, rf_val_pred, average='weighted', zero_division=0)

                # Gradient Boosting
                gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                gb_model.fit(X_train_split, y_train_split)
                gb_val_pred = gb_model.predict(X_val)
                gb_accuracy = accuracy_score(y_val, gb_val_pred)
                gb_precision = precision_score(y_val, gb_val_pred, average='weighted', zero_division=0)
                gb_recall = recall_score(y_val, gb_val_pred, average='weighted', zero_division=0)
                gb_f1 = f1_score(y_val, gb_val_pred, average='weighted', zero_division=0)

                # Save predictions on the test data
                rf_test_pred = rf_model.predict(X_test)
                gb_test_pred = gb_model.predict(X_test)

                # Store prediction results in original test dataframe
                results_df = original_test_df.copy()
                results_df['RF_Prediction'] = rf_test_pred
                results_df['GB_Prediction'] = gb_test_pred

                # Save predictions to CSV
                result_file = f"results_{os.path.basename(self.test_file)}"
                results_df.to_csv(result_file, index=False)

                # Save models and results for comparison
                self.results["Random Forest"] = {
                    "model": rf_model,
                    "test_pred": rf_test_pred,
                    "accuracy": rf_accuracy,
                    "precision": rf_precision,
                    "recall": rf_recall,
                    "f1": rf_f1
                }

                self.results["Gradient Boosting"] = {
                    "model": gb_model,
                    "test_pred": gb_test_pred,
                    "accuracy": gb_accuracy,
                    "precision": gb_precision,
                    "recall": gb_recall,
                    "f1": gb_f1
                }

                messagebox.showinfo("Model Evaluation",
                                    f"Random Forest Metrics:\n"
                                    f"  Accuracy: {rf_accuracy:.4f}\n"
                                    f"  Precision: {rf_precision:.4f}\n"
                                    f"  Recall: {rf_recall:.4f}\n"
                                    f"  F1 Score: {rf_f1:.4f}\n\n"
                                    f"Gradient Boosting Metrics:\n"
                                    f"  Accuracy: {gb_accuracy:.4f}\n"
                                    f"  Precision: {gb_precision:.4f}\n"
                                    f"  Recall: {gb_recall:.4f}\n"
                                    f"  F1 Score: {gb_f1:.4f}\n\n"
                                    f"Results saved to {result_file}")

            elif selected_model == "K-Means Clustering + Principal Component Analysis (PCA)":
                # Reduce dimensions with PCA
                pca = PCA(n_components=min(5, X_train.shape[1]))
                X_train_pca = pca.fit_transform(X_train_split)
                X_val_pca = pca.transform(X_val)
                X_test_pca = pca.transform(X_test)

                # Apply K-Means clustering
                unique_classes = len(np.unique(y_train))
                kmeans = KMeans(n_clusters=unique_classes, random_state=42)
                kmeans.fit(X_train_pca)
                kmeans_val_pred = kmeans.predict(X_val_pca)

                # Map cluster labels to original class labels
                # This is a simplistic approach; in practice, you'd need a more sophisticated mapping
                cluster_to_label = {}
                for cluster in range(unique_classes):
                    mask = kmeans.labels_ == cluster
                    if any(mask):
                        most_common = np.bincount(y_train_split[mask].astype(int)).argmax()
                        cluster_to_label[cluster] = most_common

                # Convert cluster predictions to class predictions
                kmeans_val_pred_mapped = np.array([cluster_to_label.get(label, 0) for label in kmeans_val_pred])

                # Calculate metrics
                km_accuracy = accuracy_score(y_val, kmeans_val_pred_mapped)
                km_precision = precision_score(y_val, kmeans_val_pred_mapped, average='weighted', zero_division=0)
                km_recall = recall_score(y_val, kmeans_val_pred_mapped, average='weighted', zero_division=0)
                km_f1 = f1_score(y_val, kmeans_val_pred_mapped, average='weighted', zero_division=0)

                # Predictions for test data
                kmeans_test_pred = kmeans.predict(X_test_pca)
                kmeans_test_pred_mapped = np.array([cluster_to_label.get(label, 0) for label in kmeans_test_pred])

                # Store results in original test dataframe
                results_df = original_test_df.copy()
                results_df['KMeans_PCA_Prediction'] = kmeans_test_pred_mapped

                # Save predictions to CSV
                result_file = f"results_{os.path.basename(self.test_file)}"
                results_df.to_csv(result_file, index=False)

                # Save results
                self.results["K-Means + PCA"] = {
                    "model": (kmeans, pca, cluster_to_label),
                    "test_pred": kmeans_test_pred_mapped,
                    "accuracy": km_accuracy,
                    "precision": km_precision,
                    "recall": km_recall,
                    "f1": km_f1
                }

                messagebox.showinfo("Model Evaluation",
                                    f"K-Means + PCA Metrics:\n"
                                    f"  Accuracy: {km_accuracy:.4f}\n"
                                    f"  Precision: {km_precision:.4f}\n"
                                    f"  Recall: {km_recall:.4f}\n"
                                    f"  F1 Score: {km_f1:.4f}\n\n"
                                    f"Results saved to {result_file}")

            elif selected_model == "Support Vector Machine (SVM)":
                # Train SVM model
                svm_model = SVC(kernel='rbf', probability=True, random_state=42)
                svm_model.fit(X_train_split, y_train_split)
                svm_val_pred = svm_model.predict(X_val)

                # Calculate metrics
                svm_accuracy = accuracy_score(y_val, svm_val_pred)
                svm_precision = precision_score(y_val, svm_val_pred, average='weighted', zero_division=0)
                svm_recall = recall_score(y_val, svm_val_pred, average='weighted', zero_division=0)
                svm_f1 = f1_score(y_val, svm_val_pred, average='weighted', zero_division=0)

                # Predictions for test data
                svm_test_pred = svm_model.predict(X_test)

                # Store results in original test dataframe
                results_df = original_test_df.copy()
                results_df['SVM_Prediction'] = svm_test_pred

                # Save predictions to CSV
                result_file = f"results_{os.path.basename(self.test_file)}"
                results_df.to_csv(result_file, index=False)

                # Save results
                self.results["SVM"] = {
                    "model": svm_model,
                    "test_pred": svm_test_pred,
                    "accuracy": svm_accuracy,
                    "precision": svm_precision,
                    "recall": svm_recall,
                    "f1": svm_f1
                }

                messagebox.showinfo("Model Evaluation",
                                    f"Support Vector Machine Metrics:\n"
                                    f"  Accuracy: {svm_accuracy:.4f}\n"
                                    f"  Precision: {svm_precision:.4f}\n"
                                    f"  Recall: {svm_recall:.4f}\n"
                                    f"  F1 Score: {svm_f1:.4f}\n\n"
                                    f"Results saved to {result_file}")

            elif selected_model == "Decision Tree + Logistic Regression":
                # Decision Tree
                dt_model = DecisionTreeClassifier(random_state=42)
                dt_model.fit(X_train_split, y_train_split)
                dt_val_pred = dt_model.predict(X_val)

                # Calculate metrics for Decision Tree
                dt_accuracy = accuracy_score(y_val, dt_val_pred)
                dt_precision = precision_score(y_val, dt_val_pred, average='weighted', zero_division=0)
                dt_recall = recall_score(y_val, dt_val_pred, average='weighted', zero_division=0)
                dt_f1 = f1_score(y_val, dt_val_pred, average='weighted', zero_division=0)

                # Logistic Regression
                lr_model = LogisticRegression(max_iter=1000, random_state=42)
                lr_model.fit(X_train_split, y_train_split)
                lr_val_pred = lr_model.predict(X_val)

                # Calculate metrics for Logistic Regression
                lr_accuracy = accuracy_score(y_val, lr_val_pred)
                lr_precision = precision_score(y_val, lr_val_pred, average='weighted', zero_division=0)
                lr_recall = recall_score(y_val, lr_val_pred, average='weighted', zero_division=0)
                lr_f1 = f1_score(y_val, lr_val_pred, average='weighted', zero_division=0)

                # Predictions for test data
                dt_test_pred = dt_model.predict(X_test)
                lr_test_pred = lr_model.predict(X_test)

                # Store results in original test dataframe
                results_df = original_test_df.copy()
                results_df['DT_Prediction'] = dt_test_pred
                results_df['LR_Prediction'] = lr_test_pred

                # Save predictions to CSV
                result_file = f"results_{os.path.basename(self.test_file)}"
                results_df.to_csv(result_file, index=False)

                # Save results
                self.results["Decision Tree"] = {
                    "model": dt_model,
                    "test_pred": dt_test_pred,
                    "accuracy": dt_accuracy,
                    "precision": dt_precision,
                    "recall": dt_recall,
                    "f1": dt_f1
                }

                self.results["Logistic Regression"] = {
                    "model": lr_model,
                    "test_pred": lr_test_pred,
                    "accuracy": lr_accuracy,
                    "precision": lr_precision,
                    "recall": lr_recall,
                    "f1": lr_f1
                }

                messagebox.showinfo("Model Evaluation",
                                    f"Decision Tree Metrics:\n"
                                    f"  Accuracy: {dt_accuracy:.4f}\n"
                                    f"  Precision: {dt_precision:.4f}\n"
                                    f"  Recall: {dt_recall:.4f}\n"
                                    f"  F1 Score: {dt_f1:.4f}\n\n"
                                    f"Logistic Regression Metrics:\n"
                                    f"  Accuracy: {lr_accuracy:.4f}\n"
                                    f"  Precision: {lr_precision:.4f}\n"
                                    f"  Recall: {lr_recall:.4f}\n"
                                    f"  F1 Score: {lr_f1:.4f}\n\n"
                                    f"Results saved to {result_file}")

            elif selected_model == "Naive Bayes + k-Nearest Neighbors (k-NN)":
                # Naive Bayes
                nb_model = GaussianNB()
                nb_model.fit(X_train_split, y_train_split)
                nb_val_pred = nb_model.predict(X_val)

                # Calculate metrics for Naive Bayes
                nb_accuracy = accuracy_score(y_val, nb_val_pred)
                nb_precision = precision_score(y_val, nb_val_pred, average='weighted', zero_division=0)
                nb_recall = recall_score(y_val, nb_val_pred, average='weighted', zero_division=0)
                nb_f1 = f1_score(y_val, nb_val_pred, average='weighted', zero_division=0)

                # k-NN
                knn_model = KNeighborsClassifier(n_neighbors=5)
                knn_model.fit(X_train_split, y_train_split)
                knn_val_pred = knn_model.predict(X_val)

                # Calculate metrics for k-NN
                knn_accuracy = accuracy_score(y_val, knn_val_pred)
                knn_precision = precision_score(y_val, knn_val_pred, average='weighted', zero_division=0)
                knn_recall = recall_score(y_val, knn_val_pred, average='weighted', zero_division=0)
                knn_f1 = f1_score(y_val, knn_val_pred, average='weighted', zero_division=0)

                # Predictions for test data
                nb_test_pred = nb_model.predict(X_test)
                knn_test_pred = knn_model.predict(X_test)

                # Store results in original test dataframe
                results_df = original_test_df.copy()
                results_df['NB_Prediction'] = nb_test_pred
                results_df['KNN_Prediction'] = knn_test_pred

                # Save predictions to CSV
                result_file = f"results_{os.path.basename(self.test_file)}"
                results_df.to_csv(result_file, index=False)

                # Save results
                self.results["Naive Bayes"] = {
                    "model": nb_model,
                    "test_pred": nb_test_pred,
                    "accuracy": nb_accuracy,
                    "precision": nb_precision,
                    "recall": nb_recall,
                    "f1": nb_f1
                }

                self.results["k-NN"] = {
                    "model": knn_model,
                    "test_pred": knn_test_pred,
                    "accuracy": knn_accuracy,
                    "precision": knn_precision,
                    "recall": knn_recall,
                    "f1": knn_f1
                }

                messagebox.showinfo("Model Evaluation",
                                    f"Naive Bayes Metrics:\n"
                                    f"  Accuracy: {nb_accuracy:.4f}\n"
                                    f"  Precision: {nb_precision:.4f}\n"
                                    f"  Recall: {nb_recall:.4f}\n"
                                    f"  F1 Score: {nb_f1:.4f}\n\n"
                                    f"k-NN Metrics:\n"
                                    f"  Accuracy: {knn_accuracy:.4f}\n"
                                    f"  Precision: {knn_precision:.4f}\n"
                                    f"  Recall: {knn_recall:.4f}\n"
                                    f"  F1 Score: {knn_f1:.4f}\n\n"
                                    f"Results saved to {result_file}")

            elif selected_model == "Neural Network":
                # Neural Network
                nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
                nn_model.fit(X_train_split, y_train_split)
                nn_val_pred = nn_model.predict(X_val)

                # Calculate metrics
                nn_accuracy = accuracy_score(y_val, nn_val_pred)
                nn_precision = precision_score(y_val, nn_val_pred, average='weighted', zero_division=0)
                nn_recall = recall_score(y_val, nn_val_pred, average='weighted', zero_division=0)
                nn_f1 = f1_score(y_val, nn_val_pred, average='weighted', zero_division=0)

                # Predictions for test data
                nn_test_pred = nn_model.predict(X_test)

                # Store prediction results in original test dataframe
                results_df = original_test_df.copy()
                results_df['NN_Prediction'] = nn_test_pred

                # Save predictions to CSV
                result_file = f"results_{os.path.basename(self.test_file)}"
                results_df.to_csv(result_file, index=False)

                # Save results
                self.results["Neural Network"] = {
                    "model": nn_model,
                    "test_pred": nn_test_pred,
                    "accuracy": nn_accuracy,
                    "precision": nn_precision,
                    "recall": nn_recall,
                    "f1": nn_f1
                }

                messagebox.showinfo("Model Evaluation",
                                    f"Neural Network Metrics:\n"
                                    f"  Accuracy: {nn_accuracy:.4f}\n"
                                    f"  Precision: {nn_precision:.4f}\n"
                                    f"  Recall: {nn_recall:.4f}\n"
                                    f"  F1 Score: {nn_f1:.4f}\n\n"
                                    f"Results saved to {result_file}")

            self.status_var.set(f"Model {selected_model} trained successfully")

        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
            self.status_var.set("Error: " + str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred: {str(e)}")
            self.status_var.set("Error occurred")

    def compare_models(self):
        if not self.results:
            messagebox.showerror("Error", "No models have been trained yet.")
            return

        # Create a comparison window
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Model Comparison")
        compare_window.geometry("600x400")

        # Create a frame for the comparison table
        frame = ttk.Frame(compare_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create a treeview widget
        columns = ("Model", "Accuracy", "Precision", "Recall", "F1 Score")
        tree = ttk.Treeview(frame, columns=columns, show="headings")

        # Define headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)

        # Insert data
        for model_name, results in self.results.items():
            tree.insert("", tk.END, values=(
                model_name,
                f"{results['accuracy']:.4f}",
                f"{results['precision']:.4f}",
                f"{results['recall']:.4f}",
                f"{results['f1']:.4f}"
            ))

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Find the best model based on F1 score
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        best_model_name = best_model[0]
        best_model_metrics = best_model[1]

        # Display the best model information
        best_model_frame = ttk.LabelFrame(compare_window, text="Best Model", padding=10)
        best_model_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(best_model_frame,
                  text=f"Best Model: {best_model_name}\n"
                       f"Accuracy: {best_model_metrics['accuracy']:.4f}\n"
                       f"Precision: {best_model_metrics['precision']:.4f}\n"
                       f"Recall: {best_model_metrics['recall']:.4f}\n"
                       f"F1 Score: {best_model_metrics['f1']:.4f}",
                  justify=tk.LEFT).pack(padx=10, pady=10)

    def view_prediction_results(self):
        """Open and display the prediction results"""
        if not self.test_file:
            messagebox.showerror("Error", "No test file has been selected.")
            return

        result_file = f"results_{os.path.basename(self.test_file)}"

        try:
            # Check if results file exists
            if not os.path.exists(result_file):
                messagebox.showerror("Error", "No results file found. Please train a model first.")
                return

            # Read the results file
            results_df = pd.read_csv(result_file)

            # Create a new window to display results
            results_window = tk.Toplevel(self.root)
            results_window.title("Fault Prediction Results")
            results_window.geometry("800x600")

            # Create a frame for the table
            frame = ttk.Frame(results_window, padding=10)
            frame.pack(fill=tk.BOTH, expand=True)

            # Create a treeview widget for the table
            cols = list(results_df.columns)
            tree = ttk.Treeview(frame, columns=cols, show="headings")

            # Define headings
            for col in cols:
                tree.heading(col, text=col)
                # Adjust column width based on content
                max_width = max(len(str(col)),
                                results_df[col].astype(str).map(len).max() if len(results_df) > 0 else 10)
                tree.column(col, width=min(100, max_width * 10), anchor=tk.CENTER)

            # Insert data
            for i, row in results_df.iterrows():
                values = [str(row[col]) for col in cols]
                # Highlight faulty modules
                if any(col in str(cols) and row[col] == 1 for col in row.index if 'Prediction' in col):
                    tree.insert("", tk.END, values=values, tags=("faulty",))
                else:
                    tree.insert("", tk.END, values=values)

            # Configure tag for faulty modules
            tree.tag_configure("faulty", background="lightcoral")

            # Add scrollbars
            vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

            # Grid layout for table and scrollbars
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")

            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)

            # Add summary information
            summary_frame = ttk.LabelFrame(results_window, text="Prediction Summary", padding=10)
            summary_frame.pack(fill=tk.X, padx=10, pady=10)

            # Count predicted faults for each model
            summary_text = ""
            for col in cols:
                if "Prediction" in col:
                    fault_count = sum(results_df[col] == 1)
                    total_count = len(results_df)
                    summary_text += f"{col}: {fault_count} faulty modules out of {total_count} ({fault_count / total_count * 100:.2f}%)\n"

            ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT).pack(padx=10, pady=10)

            # Add export button
            ttk.Button(results_window, text="Export Results",
                       command=lambda: self.export_results(results_df)).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Error displaying results: {str(e)}")

    def export_results(self, results_df):
        """Export the results to a user-selected location"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                results_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting results: {str(e)}")

    # Move these lines OUTSIDE and AFTER the class definition

if __name__ =="__main__":
    root = tk.Tk()
    app= DataAnalysisApp(root)
    root.mainloop()