Local Image Classifier (Flask)

This repo provides a minimal local web app that lets you upload an image and runs a Keras model from `user_model/keras_model.h5` to predict the label (and show confidence). The frontend is intentionally minimal.

Files added
- `app.py` — Flask backend that loads the model at runtime, accepts image uploads, preprocesses them, runs prediction, and shows the result.
- `templates/index.html` — Minimal HTML upload page and result display.
- `requirements.txt` — Required Python packages (Flask, Pillow, numpy). Uncomment and pick a TensorFlow line appropriate for your machine.

Setup (Windows PowerShell)

1. Create and activate a venv (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Edit `requirements.txt` and uncomment the appropriate `tensorflow` line you want to use (or run `pip install tensorflow` separately). Then install:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python app.py
```

4. Open your browser at http://127.0.0.1:5000 and upload images.

Notes & troubleshooting
- Make sure `user_model/keras_model.h5` and `user_model/labels.txt` exist. `labels.txt` should be one label per line in the same order as the model output.
- The app tries to infer model input shape; if your model expects a very custom preprocessing (e.g., different normalization, center-cropping, or specialized channels), adapt `preprocess_image` in `app.py` accordingly.
- TensorFlow installation can be large. If you prefer, install `tensorflow` or `tensorflow-cpu` manually.

Next steps (optional)
- Add more robust preprocessing tuned to your model.
- Add example test images and automated tests.
- Add Dockerfile for reproducible runs.
