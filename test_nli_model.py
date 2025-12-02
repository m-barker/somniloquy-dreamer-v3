import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    nli_model = pipeline(
        "text-classification",
        model="facebook/bart-large-mnli",
        top_k=None,
    )

    hypothesis = "i will achieve the following exact behaviour: go to the red key and then go to the purple box"

    premise = "first i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will go to the purple key and then i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will go to the purple key and then i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will move next to the purple key and then i will move next to the red box and i will move next to the purple key and then i will go to the red key and i will then go to the purple box"

    pair = [f"{premise} </s><s> {hypothesis}"]
    result = nli_model(pair)

    print(result)
