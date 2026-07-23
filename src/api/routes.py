from fastapi import APIRouter, HTTPException

from src.detector import SensitiveDetector
from src.api.schemas import AnalyzeRequest, BatchAnalyzeRequest

router = APIRouter()

detector = SensitiveDetector()

###########################################################
# Health
###########################################################


@router.get("/health")
def health():

    return {
        "status": "ok",
        "model": "SensitiveAI",
    }


###########################################################
# Analyze
###########################################################


@router.post("/analyze")
def analyze(request: AnalyzeRequest):

    try:

        return detector.analyze(request.text)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


###########################################################
# Explain
###########################################################


@router.post("/explain")
def explain(request: AnalyzeRequest):

    return detector.explain(request.text)


###########################################################
# Predict
###########################################################


@router.post("/predict")
def predict(request: AnalyzeRequest):

    return {
        "decision": detector.predict(request.text),
    }


###########################################################
# Score
###########################################################


@router.post("/score")
def score(request: AnalyzeRequest):

    return {
        "score": detector.score(request.text),
    }


###########################################################
# Categories
###########################################################


@router.post("/categories")
def categories(request: AnalyzeRequest):

    return {
        "categories": detector.categories(request.text),
    }


###########################################################
# Batch
###########################################################


@router.post("/batch")
def batch(request: BatchAnalyzeRequest):

    results = []

    for text in request.texts:

        results.append(detector.analyze(text))

    return {
        "count": len(results),
        "results": results,
    }
