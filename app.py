from flask import Flask, jsonify, render_template, request
from router import load_model, predict


def create_app(model=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
    active_model = model if model is not None else load_model()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=True)

    @app.post("/api/predict")
    def classify():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object with a text field."), 400
        try:
            return jsonify(predict(active_model, payload.get("text")))
        except ValueError as error:
            return jsonify(error=str(error)), 400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error="Request body is too large."), 413

    return app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5003, debug=False)
