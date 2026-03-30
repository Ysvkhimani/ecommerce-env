from pydantic import BaseModel
from typing import List


# State Model
class State(BaseModel):
    cart: List[str]
    total: float
    coupon_applied: bool
    payment_done: bool
    order_status: str


# Action Model
class Action(BaseModel):
    action: str