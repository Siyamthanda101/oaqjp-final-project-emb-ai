"""Flask web server for the emotion detection application."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector


app = Flask(__name__, template_folder="../templates", static_folder="../static")


@app.route("/")
def home():
    """Render the application's home page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route():
    """Analyze submitted text and return its emotion scores."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    emotions = emotion_detector(text_to_analyze)
    if emotions["dominant_emotion"] is None:
        return "Invalid text! Please try again!"

    return (
        "For the given statement, the system response is "
        f"'anger': {emotions['anger']}, "
        f"'disgust': {emotions['disgust']}, "
        f"'fear': {emotions['fear']}, "
        f"'joy': {emotions['joy']} and "
        f"'sadness': {emotions['sadness']}. "
        f"The dominant emotion is <strong>{emotions['dominant_emotion']}</strong>."
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
