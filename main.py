import flet as ft
import numpy as np
import pandas as pd
import io

# ==========================================
# YOUR CUSTOM NEURAL NETWORK SCRIPT (WITH SMITH MATCHING IN BACKPROP)
# ==========================================
def smith_map(x):
    return x / (1.0 + np.abs(x))


def smith_match_signal(signal):
    return signal / (1.0 + np.abs(signal))


def bounded_softmax(matrix_z):
    shift_z = matrix_z - np.max(matrix_z, axis=1, keepdims=True)
    exp_matrix = np.exp(shift_z)
    return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)


class CustomDistanceLayer:
    def __init__(self, input_dim, output_dim, is_output_layer=False):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros((1, output_dim))
        self.is_output_layer = is_output_layer
        self.last_input = None
        self.last_z = None
        self.last_activated = None

    def forward(self, X):
        self.last_input = X
        self.last_z = np.dot(X, self.W) + self.b
        if self.is_output_layer:
            mapped_z = smith_map(self.last_z)
            self.last_activated = bounded_softmax(mapped_z)
        else:
            self.last_activated = smith_map(self.last_z)
        return self.last_activated

    def tweak_weights_by_distance(self, layer_error_direction, learning_rate):
        smith_error = smith_match_signal(layer_error_direction)
        weight_update = np.dot(self.last_input.T, smith_error) / max(self.last_input.shape[0], 1)
        bias_update = np.sum(smith_error, axis=0, keepdims=True) / max(self.last_input.shape[0], 1)
        self.W -= learning_rate * weight_update
        self.b -= learning_rate * bias_update
        return np.dot(smith_error, self.W.T)


class MultiCategoryNetwork:
    def __init__(self, layers_dim):
        self.layers = []
        for i in range(len(layers_dim) - 2):
            self.layers.append(CustomDistanceLayer(layers_dim[i], layers_dim[i + 1], is_output_layer=False))
        self.layers.append(CustomDistanceLayer(layers_dim[-2], layers_dim[-1], is_output_layer=True))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train_step(self, X, Y_target, learning_rate):
        predictions = self.forward(X)
        shortest_distance_error = predictions - Y_target
        current_error_vector = smith_match_signal(shortest_distance_error)
        for layer in reversed(self.layers):
            current_error_vector = layer.tweak_weights_by_distance(current_error_vector, learning_rate)
        return -np.mean(np.sum(Y_target * np.log(predictions + 1e-15), axis=1))

# ==========================================
# MOBILE INTERFACE ENVIRONMENT (FLET)
# ==========================================
def main(page: ft.Page):
    page.title = "Local NN Trainer"
    page.theme_mode = ft.ThemeMode.DARK
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20
    page.bgcolor = ft.Colors.SURFACE

    uploaded_df = None

    dataset_summary = ft.Text("No dataset loaded yet.", size=13, color=ft.Colors.GREY_400)
    status_text = ft.Text("Upload a CSV file to get started.", color=ft.Colors.GREY_400)
    run_status = ft.Text("Ready to train.", size=13, color=ft.Colors.GREEN_300)
    progress_bar = ft.ProgressBar(value=0, color=ft.Colors.GREEN_400, bgcolor=ft.Colors.BLUE_GREY_900)
    log_output = ft.TextField(label="Console Output", multiline=True, min_lines=10, max_lines=12, read_only=True, text_size=12)
    lr_slider = ft.Slider(min=0.001, max=0.5, value=0.1, divisions=50, label="LR: {value}")

    input_dropdown = ft.Dropdown(label="Primary Input Feature Column", options=[])
    feature_text = ft.TextField(label="Feature columns (comma-separated, optional)", hint_text="Use all numeric columns or list names like A, B, C")
    output_dropdown = ft.Dropdown(label="Target Label Column", options=[])

    def show_prediction_dialog(nn, feature_columns, class_names):
        value_fields = [
            ft.TextField(
                label=name,
                keyboard_type=ft.KeyboardType.NUMBER,
                autofocus=index == 0,
            )
            for index, name in enumerate(feature_columns)
        ]
        prediction_text = ft.Text()

        def predict_sample(e):
            try:
                values = [float(field.value) for field in value_fields]
                if not np.all(np.isfinite(values)):
                    raise ValueError("Enter a finite number for every feature.")

                probabilities = nn.forward(np.asarray([values], dtype=float))[0]
                predicted_index = int(np.argmax(probabilities))
                prediction_text.value = (
                    f"Prediction: {class_names[predicted_index]} "
                    f"({probabilities[predicted_index] * 100:.1f}% confidence)"
                )
                prediction_text.color = ft.Colors.GREEN_300
            except (TypeError, ValueError) as ex:
                prediction_text.value = str(ex) or "Enter a valid number for every feature."
                prediction_text.color = ft.Colors.RED_300
            page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Try the trained model"),
            content=ft.Column(
                [
                    ft.Text("Enter values for the trained feature columns:"),
                    *value_fields,
                    prediction_text,
                ],
                tight=True,
                scroll=ft.ScrollMode.AUTO,
            ),
            actions=[
                ft.TextButton("Predict", on_click=predict_sample),
                ft.TextButton("Close", on_click=lambda e: page.pop_dialog()),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.show_dialog(dialog)

    def load_csv_file(file):
        nonlocal uploaded_df
        try:
            if file.bytes is not None:
                df = pd.read_csv(io.BytesIO(file.bytes))
            elif file.path:
                df = pd.read_csv(file.path)
            else:
                raise ValueError("The selected CSV could not be read. Please select it again.")

            if df.empty:
                raise ValueError("The CSV file is empty.")

            numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_columns:
                raise ValueError("The CSV must contain at least one numeric feature column.")

            uploaded_df = df
            input_dropdown.options = [ft.DropdownOption(col) for col in numeric_columns]
            output_dropdown.options = [ft.DropdownOption(col) for col in df.columns]

            input_dropdown.value = numeric_columns[0]
            output_dropdown.value = next((col for col in df.columns if col != input_dropdown.value), df.columns[0])
            feature_text.value = ", ".join(numeric_columns)

            status_text.value = f"Loaded: {file.name} ({len(df)} rows, {len(df.columns)} columns)."
            dataset_summary.value = f"Rows: {len(df)} | Columns: {len(df.columns)} | Numeric: {len(numeric_columns)}"
            run_status.value = "Dataset ready. Select features and train."
            run_status.color = ft.Colors.BLUE_200
            page.update()
        except Exception as ex:
            status_text.value = f"Error reading file: {str(ex)}"
            dataset_summary.value = "No dataset loaded yet."
            uploaded_df = None
            run_status.value = "Error loading CSV."
            run_status.color = ft.Colors.RED_300
            page.update()

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def pick_csv(e):
        try:
            files = await file_picker.pick_files(
                dialog_title="Select a CSV dataset",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["csv"],
                allow_multiple=False,
                with_data=True,
            )
            if files:
                load_csv_file(files[0])
        except Exception as ex:
            status_text.value = f"Error selecting CSV: {ex}"
            run_status.value = "CSV selection failed."
            run_status.color = ft.Colors.RED_300
            page.update()

    def run_training_loop(e):
        nonlocal uploaded_df

        if uploaded_df is None:
            log_output.value = "Please upload a CSV file before running training."
            run_status.value = "Training blocked: no dataset loaded."
            run_status.color = ft.Colors.RED_300
            page.update()
            return

        target_col = output_dropdown.value
        if target_col is None:
            log_output.value = "Select a target column."
            run_status.value = "Training blocked: target column missing."
            run_status.color = ft.Colors.RED_300
            page.update()
            return

        raw_feature_names = feature_text.value.strip()
        if raw_feature_names:
            feature_columns = [name.strip() for name in raw_feature_names.split(',') if name.strip()]
        else:
            feature_columns = [input_dropdown.value] if input_dropdown.value else []

        if not feature_columns:
            log_output.value = "Choose at least one feature column."
            run_status.value = "Training blocked: no feature columns selected."
            run_status.color = ft.Colors.RED_300
            page.update()
            return

        missing_cols = [name for name in feature_columns if name not in uploaded_df.columns]
        if missing_cols:
            log_output.value = f"Feature columns not found: {missing_cols}."
            run_status.value = "Training blocked: missing feature columns."
            run_status.color = ft.Colors.RED_300
            page.update()
            return

        if target_col in feature_columns:
            log_output.value = "Target column must be different from the feature columns."
            run_status.value = "Training blocked: target overlaps features."
            run_status.color = ft.Colors.RED_300
            page.update()
            return

        try:
            X_train = uploaded_df[feature_columns].to_numpy(dtype=float)
            target_series = uploaded_df[target_col]
            unique_targets = pd.Series(target_series).astype(str).unique().tolist()

            if len(unique_targets) < 2:
                raise ValueError("The target column must contain at least two distinct classes.")

            target_to_index = {label: idx for idx, label in enumerate(unique_targets)}
            y_indices = pd.Series(target_series).astype(str).map(target_to_index).to_numpy(dtype=int)
            Y_train = np.eye(len(unique_targets), dtype=float)[y_indices]

            network_dimensions = [X_train.shape[1], max(4, X_train.shape[1] * 2), len(unique_targets)]
            nn = MultiCategoryNetwork(network_dimensions)

            selected_lr = lr_slider.value
            run_status.value = "Training in progress..."
            run_status.color = ft.Colors.ORANGE_300
            progress_bar.value = 0.0
            log_output.value = "--- Initializing Multi-Category Network from uploaded CSV ---\n"
            log_output.value += f"Feature columns: {feature_columns}\nTarget column: {target_col}\nClasses: {unique_targets}\n"
            log_output.value += f"Using learning rate: {selected_lr}\n"
            page.update()

            total_epochs = 600
            for epoch in range(1, total_epochs + 1):
                current_loss = nn.train_step(X_train, Y_train, learning_rate=selected_lr)
                if epoch % 100 == 0 or epoch == 1:
                    log_output.value += f"Cycle {epoch:3d} | Total Categorical Loss: {current_loss:.6f}\n"
                progress_bar.value = epoch / total_epochs
                if epoch % 50 == 0:
                    page.update()
            progress_bar.value = 1.0

            final_outputs = nn.forward(X_train)
            accuracy = np.mean(np.argmax(final_outputs, axis=1) == np.argmax(Y_train, axis=1))

            log_output.value += "\n--- Final Probability Display Verification ---\n"
            for i in range(len(X_train)):
                percentages = [f"{p * 100:.1f}%" for p in final_outputs[i]]
                predicted_class = unique_targets[np.argmax(final_outputs[i])]
                target_value = str(target_series.iloc[i])
                feature_values = ", ".join(f"{name}={X_train[i, idx]}" for idx, name in enumerate(feature_columns))
                log_output.value += f"Sample {i}: {feature_values} -> {percentages} | Predicted: {predicted_class} | Target: {target_value}\n"

            log_output.value += f"\nTraining Accuracy: {accuracy * 100:.2f}%\n"
            run_status.value = f"Training complete. Accuracy: {accuracy * 100:.2f}%"
            run_status.color = ft.Colors.GREEN_300
            page.update()
            show_prediction_dialog(nn, feature_columns, unique_targets)
        except Exception as ex:
            log_output.value = f"Training error: {str(ex)}\n"
            run_status.value = "Training failed."
            run_status.color = ft.Colors.RED_300
            page.update()

    page.add(
        ft.AppBar(
            title=ft.Text("On-Device NN Trainer", size=22, weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            color=ft.Colors.ON_SURFACE,
        ),
        ft.Container(
            content=ft.Column([
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Dataset", size=20, weight=ft.FontWeight.BOLD),
                            ft.Row([
                                ft.FilledButton("📁 Upload CSV Dataset", icon=ft.Icons.UPLOAD_FILE, on_click=pick_csv),
                                status_text,
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            dataset_summary,
                        ], spacing=8),
                        padding=18,
                        border_radius=16,
                    ),
                    margin=0,
                ),
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Model Setup", size=20, weight=ft.FontWeight.BOLD),
                            input_dropdown,
                            feature_text,
                            output_dropdown,
                        ], spacing=12),
                        padding=18,
                        border_radius=16,
                    ),
                    margin=0,
                ),
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Training", size=20, weight=ft.FontWeight.BOLD),
                            ft.Text("Adjust Starting Learning Rate:"),
                            lr_slider,
                            run_status,
                            progress_bar,
                            ft.FilledButton(
                                "🚀 Run Script Locally On Mobile",
                                style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE),
                                on_click=run_training_loop,
                            ),
                        ], spacing=12),
                        padding=18,
                        border_radius=16,
                    ),
                    margin=0,
                ),
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Training Log", size=20, weight=ft.FontWeight.BOLD),
                            log_output,
                        ], spacing=10),
                        padding=18,
                        border_radius=16,
                    ),
                    margin=0,
                ),
            ], spacing=18),
            padding=0,
        )
    )

if hasattr(ft, "run"):
    ft.run(main)
else:
    ft.app(target=main)