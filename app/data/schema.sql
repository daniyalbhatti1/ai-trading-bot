CREATE TABLE IF NOT EXISTS candles (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  symbol TEXT NOT NULL,
  open REAL, 
  high REAL, 
  low REAL, 
  close REAL, 
  volume REAL,
  UNIQUE(ts, symbol)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles(symbol, ts);

CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT CHECK(side IN ('LONG','SHORT','FLAT')) NOT NULL,
  confidence REAL,
  reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts DESC);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT CHECK(side IN ('BUY','SELL')) NOT NULL,
  qty REAL NOT NULL,
  type TEXT,
  limit_price REAL,
  stop_price REAL,
  client_order_id TEXT,
  status TEXT,
  fill_price REAL,
  fill_ts TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts DESC);

CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY,
  symbol TEXT UNIQUE NOT NULL,
  avg_price REAL,
  qty REAL,
  unrealized_pl REAL,
  last_update TEXT,
  entry_qty REAL,              -- Original entry quantity
  first_tp_hit INTEGER DEFAULT 0,  -- 1 if first TP was hit
  stop_loss_pct REAL           -- Current stop loss percentage (may be moved to breakeven)
);

CREATE TABLE IF NOT EXISTS equity_curve (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  equity REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts DESC);

-- Trade Journal: Complete record of all trades with context and analysis
CREATE TABLE IF NOT EXISTS trade_journal (
  id INTEGER PRIMARY KEY,
  
  -- Basic Trade Info
  entry_time TEXT NOT NULL,
  exit_time TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT CHECK(side IN ('LONG','SHORT')) NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL NOT NULL,
  qty REAL NOT NULL,
  pnl REAL NOT NULL,
  pnl_pct REAL NOT NULL,
  exit_reason TEXT,
  
  -- Trade Duration
  duration_minutes INTEGER,
  
  -- Entry Context (Market Conditions at Entry)
  entry_rsi REAL,
  entry_macd REAL,
  entry_ema_fast REAL,
  entry_ema_slow REAL,
  entry_atr REAL,
  entry_volume REAL,
  
  -- ICT Context at Entry
  entry_liq_sweep_bull INTEGER,
  entry_liq_sweep_bear INTEGER,
  entry_bos_bull INTEGER,
  entry_bos_bear INTEGER,
  entry_fvg_bull INTEGER,
  entry_fvg_bear INTEGER,
  entry_retrace_to_fvg_bull INTEGER,
  entry_retrace_to_fvg_bear INTEGER,
  entry_engulfing_bull INTEGER,
  entry_engulfing_bear INTEGER,
  entry_ict_setup_bull INTEGER,
  entry_ict_setup_bear INTEGER,
  
  -- Key Levels at Entry
  entry_pdh REAL,  -- Previous Day High
  entry_pdl REAL,  -- Previous Day Low
  entry_4h_high REAL,
  entry_4h_low REAL,
  entry_1h_high REAL,
  entry_1h_low REAL,
  
  -- Exit Context (Market Conditions at Exit)
  exit_rsi REAL,
  exit_macd REAL,
  exit_ema_fast REAL,
  exit_ema_slow REAL,
  exit_atr REAL,
  
  -- Trade Performance Metrics
  max_favorable_excursion REAL,  -- Best unrealized profit during trade
  max_adverse_excursion REAL,    -- Worst unrealized loss during trade
  risk_reward_ratio REAL,
  
  -- ML Model Context
  ml_confidence REAL,
  ml_prediction TEXT,  -- What the ML model predicted
  
  -- Trade Quality Score (calculated post-trade)
  quality_score REAL,  -- 0-100 score based on setup quality and execution
  
  -- Learning Notes
  what_worked TEXT,    -- Analysis of why trade succeeded
  what_failed TEXT,    -- Analysis of why trade failed
  lessons_learned TEXT,
  
  -- Timestamps
  created_at TEXT NOT NULL,
  analyzed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_trade_journal_symbol ON trade_journal(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_journal_entry_time ON trade_journal(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_journal_pnl ON trade_journal(pnl);
CREATE INDEX IF NOT EXISTS idx_trade_journal_side ON trade_journal(side);

-- Trade Analysis: Aggregated insights and patterns
CREATE TABLE IF NOT EXISTS trade_analysis (
  id INTEGER PRIMARY KEY,
  
  -- Pattern Identification
  pattern_name TEXT NOT NULL,  -- e.g., "ICT Bullish Setup", "Liquidity Sweep + BOS"
  pattern_description TEXT,
  
  -- Performance Stats
  total_trades INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  losing_trades INTEGER DEFAULT 0,
  win_rate REAL,
  avg_pnl REAL,
  avg_win REAL,
  avg_loss REAL,
  profit_factor REAL,
  
  -- Best Conditions for This Pattern
  best_rsi_range_min REAL,
  best_rsi_range_max REAL,
  best_time_of_day TEXT,
  best_symbols TEXT,  -- JSON array of symbols where pattern works best
  
  -- Last Updated
  last_updated TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_analysis_pattern ON trade_analysis(pattern_name);
CREATE INDEX IF NOT EXISTS idx_trade_analysis_win_rate ON trade_analysis(win_rate DESC);

-- Model Training History: Track model retraining from trade experience
CREATE TABLE IF NOT EXISTS model_training_history (
  id INTEGER PRIMARY KEY,
  
  -- Training Info
  training_date TEXT NOT NULL,
  model_version TEXT NOT NULL,
  trades_used INTEGER,  -- Number of trades used for training
  
  -- Performance Before/After
  win_rate_before REAL,
  win_rate_after REAL,
  profit_factor_before REAL,
  profit_factor_after REAL,
  
  -- Model Metrics
  accuracy REAL,
  precision_long REAL,
  precision_short REAL,
  recall_long REAL,
  recall_short REAL,
  f1_score REAL,
  
  -- Training Parameters
  training_params TEXT,  -- JSON of hyperparameters used
  
  -- Notes
  improvements TEXT,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_training_date ON model_training_history(training_date DESC);

