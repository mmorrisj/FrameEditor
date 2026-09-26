"""Web entry point (kept for the systemd unit): same as `python -m framekit serve`."""
from framekit.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, threaded=True)
