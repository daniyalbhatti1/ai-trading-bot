"""API route handlers."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from app.trading.broker_alpaca import Broker
from app.data.repos import get_recent_signals, get_recent_orders, get_open_positions, get_equity_curve
from app.core.logger import logger

router = APIRouter()

# Global state for trading control
trading_state = {
    'running': False,
    'paused': False
}


class StatusResponse(BaseModel):
    status: str
    running: bool
    paused: bool
    account: Dict


class ControlRequest(BaseModel):
    action: str  # start, stop, pause, resume


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status."""
    try:
        broker = Broker()
        account_info = broker.get_account_status()
        
        return StatusResponse(
            status="ok",
            running=trading_state['running'],
            paused=trading_state['paused'],
            account=account_info
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/control")
async def control_trading(request: ControlRequest):
    """Control trading execution."""
    action = request.action.lower()
    
    if action == "start":
        trading_state['running'] = True
        trading_state['paused'] = False
        return {"message": "Trading started"}
    
    elif action == "stop":
        trading_state['running'] = False
        trading_state['paused'] = False
        return {"message": "Trading stopped"}
    
    elif action == "pause":
        trading_state['paused'] = True
        return {"message": "Trading paused"}
    
    elif action == "resume":
        trading_state['paused'] = False
        return {"message": "Trading resumed"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")


@router.get("/signals")
async def get_signals(limit: int = 50):
    """Get recent signals."""
    try:
        df = get_recent_signals(limit)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
async def get_orders(limit: int = 100):
    """Get recent orders."""
    try:
        df = get_recent_orders(limit)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions():
    """Get open positions."""
    try:
        df = get_open_positions()
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/equity")
async def get_equity(days: int = 30):
    """Get equity curve."""
    try:
        df = get_equity_curve(days)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error getting equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def is_trading_active() -> bool:
    """Check if trading is active."""
    return trading_state['running'] and not trading_state['paused']

