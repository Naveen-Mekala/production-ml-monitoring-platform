from pydantic import BaseModel, Field, ConfigDict


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    income: float = Field(gt=0, le=1_000_000)
    credit_score: int = Field(ge=300, le=850)
    debt_to_income: float = Field(ge=0, le=1)
    loan_amount: float = Field(gt=0, le=1_000_000)
    employment_years: float = Field(ge=0, le=60)
    interest_rate: float = Field(gt=0, le=100)


class PredictionResponse(BaseModel):
    prediction: int
    default_probability: float
    risk_level: str
    model_version: str
