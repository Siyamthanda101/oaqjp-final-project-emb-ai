import json
import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector
import server


class TestEmotionDetector(unittest.TestCase):
	def assert_dominant_emotion(self, statement, expected_emotion):
		emotion_names = ("anger", "disgust", "fear", "joy", "sadness")
		emotion_scores = {
			name: 0.9 if name == expected_emotion else 0.025
			for name in emotion_names
		}
		response = Mock(
			status_code=200,
			text=json.dumps({"emotionPredictions": [{"emotion": emotion_scores}]})
		)

		with patch(
			"EmotionDetection.emotion_detection.requests.post",
			return_value=response,
		) as post:
			result = emotion_detector(statement)

		self.assertEqual(result["dominant_emotion"], expected_emotion)
		self.assertEqual(
			{emotion: result[emotion] for emotion in emotion_names},
			emotion_scores,
		)
		post.assert_called_once_with(
			"https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
			headers={
				"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
			},
			json={"raw_document": {"text": statement}},
		)

	def test_joy(self):
		self.assert_dominant_emotion("I am glad this happened", "joy")

	def test_anger(self):
		self.assert_dominant_emotion("I am really mad about this", "anger")

	def test_disgust(self):
		self.assert_dominant_emotion(
			"I feel disgusted just hearing about this", "disgust"
		)

	def test_sadness(self):
		self.assert_dominant_emotion("I am so sad about this", "sadness")

	def test_fear(self):
		self.assert_dominant_emotion("I am really afraid that this will happen", "fear")

	def test_bad_request_returns_none_values(self):
		response = Mock(status_code=400)
		with patch(
			"EmotionDetection.emotion_detection.requests.post",
			return_value=response,
		) as post:
			result = emotion_detector("")

		self.assertEqual(
			result,
			{
				"anger": None,
				"disgust": None,
				"fear": None,
				"joy": None,
				"sadness": None,
				"dominant_emotion": None,
			},
		)
		post.assert_called_once_with(
			"https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
			headers={
				"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
			},
			json={"raw_document": {"text": ""}},
		)

	def test_blank_input_displays_invalid_text_message(self):
		invalid_result = {
			"anger": None,
			"disgust": None,
			"fear": None,
			"joy": None,
			"sadness": None,
			"dominant_emotion": None,
		}
		with patch("server.emotion_detector", return_value=invalid_result):
			response = server.app.test_client().get(
				"/emotionDetector", query_string={"textToAnalyze": ""}
			)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(
			response.get_data(as_text=True), "Invalid text! Please try again!"
		)


if __name__ == "__main__":
	unittest.main(verbosity=2)