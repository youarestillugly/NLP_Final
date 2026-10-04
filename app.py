from pathlib import Path

import joblib
import streamlit as st


st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)



BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "imdb_sentiment_final_pipeline.joblib"
)



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("The trained model could not be found.")
    st.code(f"Expected model location:\n{MODEL_PATH}")
    st.stop()


try:
    model = load_model()

except Exception as e:
    st.error("The trained model could not be loaded.")
    st.code(
        f"Expected model location:\n{MODEL_PATH}\n\n"
        f"Error:\n{e}"
    )
    st.stop()


st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the NLP model will predict "
    "whether the review is Positive or Negative."
)

st.info(
    "IMDB rating scale used in the dataset: "
    "Negative = 1–4, Positive = 7–10."
)



review = st.text_area(
    "Movie Review",
    placeholder=(
        "Example: The acting was excellent and "
        "I really enjoyed this movie."
    ),
    height=180
)



if st.button("Predict Sentiment", type="primary"):

    if not review.strip():

        st.warning(
            "Please enter a movie review before predicting."
        )

    elif len(review.strip()) < 3:

        st.warning(
            "Please enter a little more text so the model "
            "can make a meaningful prediction."
        )

    else:


        prediction = model.predict([review])[0]


        try:
            decision_score = float(
                model.decision_function([review])[0]
            )

        except Exception:
            decision_score = 0.0



        # Negative: 1–4
        # Positive: 7–10


        if prediction == 0:

            sentiment = "Negative"
            sentiment_emoji = "😞"

            # Convert negative decision strength
            # into a score from 1 to 4.

            strength = min(abs(decision_score), 3.0)

            score = round(
                4 - strength
            )

            score = max(1, min(4, score))

        else:

            sentiment = "Positive"
            sentiment_emoji = "😊"

            # Convert positive decision strength
            # into a score from 7 to 10.

            strength = min(abs(decision_score), 3.0)

            score = round(
                7 + strength
            )

            score = max(7, min(10, score))


        if prediction == 1:

            st.success(
                f"### {sentiment_emoji} Predicted Sentiment: "
                f"{sentiment}"
            )

        else:

            st.error(
                f"### {sentiment_emoji} Predicted Sentiment: "
                f"{sentiment}"
            )



        st.metric(
            "⭐ IMDB Score",
            f"{score}/10"
        )


        if score <= 4:

            st.caption(
                f"The review is classified as Negative "
                f"with an IMDB-style score of {score}/10."
            )

        else:

            st.caption(
                f"The review is classified as Positive "
                f"with an IMDB-style score of {score}/10."
            )

st.divider()

st.caption(
    "Model: Classical NLP pipeline using text preprocessing, "
    "TF-IDF feature extraction, and a tuned Linear SVM classifier."
)

st.caption(
    "Note: The displayed 1–10 score is an IMDB-style score "
    "derived from the binary sentiment model. It is not the "
    "original reviewer's exact IMDB rating."
)
