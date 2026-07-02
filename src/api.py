"""AMLGuard scoring API — Phase 2 skeleton.

Status: SKELETON — intentionally not implemented for the academic delivery.
The Projeto Integrador rubric covers the modelling pipeline (notebook
Sections 1-9); this file marks the serving layer planned in the repository
roadmap (see README). A minimal in-notebook demonstration of the same
``/score`` idea lives in the notebook's production-layer section.

Phase 2 plan
------------
1. Serialize the winning pipeline from the notebook with ``joblib.dump``
   (preprocessor + model together, so encoding stays consistent).
2. Implement the endpoint below and validate inputs with Pydantic.
3. Add pytest coverage for schema validation and score range.
4. Containerize with the docker-compose.yml at the repository root.

Run (once implemented)::

    uvicorn src.api:app --reload
"""

# TODO(Phase 2): uncomment when implementing.
# from pathlib import Path
#
# import joblib
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel, Field
#
# from src.features import build_model_features
#
# MODEL_PATH = Path("models/amlguard_pipeline.joblib")
#
# app = FastAPI(title="AMLGuard", version="0.1.0")
# pipeline = joblib.load(MODEL_PATH)  # preprocessor + classifier in one object
#
#
# class Transaction(BaseModel):
#     """Raw transaction in the IBM AML HI-Small schema."""
#
#     timestamp: str = Field(..., alias="Timestamp")
#     from_bank: int = Field(..., alias="From Bank")
#     account: str = Field(..., alias="Account")
#     to_bank: int = Field(..., alias="To Bank")
#     account_1: str = Field(..., alias="Account.1")
#     amount_received: float = Field(..., alias="Amount Received")
#     receiving_currency: str = Field(..., alias="Receiving Currency")
#     amount_paid: float = Field(..., alias="Amount Paid")
#     payment_currency: str = Field(..., alias="Payment Currency")
#     payment_format: str = Field(..., alias="Payment Format")
#
#
# @app.post("/score")
# def score(tx: Transaction) -> dict:
#     """Return the laundering probability for a single transaction.
#
#     NOTE: sender_previous_tx_count requires account history; the online
#     version needs a feature store / running counter per account.
#     TODO(Phase 2): decide the online strategy for stateful features.
#     """
#     frame = pd.DataFrame([tx.model_dump(by_alias=True)])
#     X = build_model_features(frame)
#     probability = float(pipeline.predict_proba(X)[0, 1])
#     return {"laundering_probability": probability}
