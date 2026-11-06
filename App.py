"""
Disegna e vIA! - server di inferenza.

Riceve il disegno dal canvas del browser, lo prepara come i bitmap di QuickDraw
(tratti bianchi su sfondo nero, centrato, quadrato, 28x28) e risponde con le
3 categorie più probabili.

Avvio:  python app.py   ->   http://localhost:5000
"""
import base64
import io
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from tensorflow import keras

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- DA ADATTARE AL TUO MODELLO --------------------------------------------
MODEL_PATH = os.path.join(BASE, "model", "modello_disegni.keras")  # oppure model.h5
LABELS_PATH = os.path.join(BASE, "model", "labels.txt")  # una categoria per riga,
#                                                          stesso ordine dell'allenamento
SCALE = 255.0         # in allenamento hai fatto x / 255 ? lascia 255.0. Altrimenti 1.0
SALVA_DEBUG = False   # True = salva debug_input.png: è esattamente ciò che vede la CNN
# -----------------------------------------------------------------------------

model = keras.models.load_model(MODEL_PATH)
shape = model.input_shape  # es. (None, 28, 28, 1) oppure (None, 784)
if len(shape) == 4:
    SIDE, FLAT = shape[1], False
else:
    SIDE, FLAT = int(round(shape[1] ** 0.5)), True

n_out = model.output_shape[-1]
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = [riga.strip() for riga in f if riga.strip()]
else:
    labels = [f"categoria {i}" for i in range(n_out)]
if len(labels) != n_out:
    raise SystemExit(
        f"labels.txt ha {len(labels)} righe ma il modello ha {n_out} uscite: controlla il file."
    )

app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="")


def prepara(data_url):
    """PNG dal canvas -> array pronto per la CNN (None se la lavagna è vuota)."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("L")
    arr = np.asarray(img)
    ys, xs = np.where(arr > 40)
    if len(xs) == 0:
        return None

    # ritaglio quadrato attorno al disegno, con un po' di margine (come QuickDraw)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    lato = int(max(x1 - x0 + 1, y1 - y0 + 1) * 1.15) + 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - lato / 2), int(cy - lato / 2), int(cx + lato / 2), int(cy + lato / 2))
    img = img.crop(box).resize((SIDE, SIDE), Image.LANCZOS)

    if SALVA_DEBUG:
        img.save(os.path.join(BASE, "debug_input.png"))

    x = np.asarray(img, dtype="float32") / SCALE
    return x.reshape(1, SIDE * SIDE) if FLAT else x.reshape(1, SIDE, SIDE, 1)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/categories")
def categories():
    return jsonify(labels)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    image = data.get("image")
    if not image:
        return jsonify(error="Immagine mancante"), 400
    try:
        x = prepara(image)
    except Exception:
        return jsonify(error="Immagine non valida"), 400
    if x is None:
        return jsonify(top=[])

    probs = model.predict(x, verbose=0)[0]
    if not np.isclose(probs.sum(), 1.0, atol=1e-3):  # uscita senza softmax
        e = np.exp(probs - probs.max())
        probs = e / e.sum()

    migliori = np.argsort(probs)[::-1][:3]
    return jsonify(top=[{"label": labels[i], "prob": float(probs[i])} for i in migliori])


if __name__ == "__main__":
    # 0.0.0.0 = raggiungibile anche da un telefono sulla stessa rete wifi
    app.run(host="0.0.0.0", port=5000, debug=False)