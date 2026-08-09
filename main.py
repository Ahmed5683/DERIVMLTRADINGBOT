import asyncio
import json
import os
import threading
import time
import warnings
from datetime import datetime
from flask import Flask, jsonify
import websockets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from talipp.indicators import MACD, StochRSI, DPO, RSI
from dotenv import load_dotenv
import requests
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

load_dotenv()

warnings.filterwarnings('ignore')

# --- Flask App for Render ---
app = Flask(__name__)

# --- Configuration from Environment Variables ---
DERIV_APP_ID = os.environ.get('DERIV_APP_ID')
DERIV_TOKEN = os.environ.get('DERIV_TOKEN')

# Validate environment variables
if not DERIV_APP_ID:
    raise Exception("❌ DERIV_APP_ID environment variable is REQUIRED for trading!")
if not DERIV_TOKEN:
    raise Exception("❌ DERIV_TOKEN environment variable is REQUIRED for trading!")

# ✅ CORRECT - Public WebSocket endpoint for market data
PUBLIC_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"

# Trading Parameters - All configurable via environment variables
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.60'))
BASE_STAKE = float(os.environ.get('BASE_STAKE', '1.0'))
TP_MULTIPLIER = float(os.environ.get('TP_MULTIPLIER', '1.5'))
SL_MULTIPLIER = float(os.environ.get('SL_MULTIPLIER', '0.45'))
ENABLE_TRADING = os.environ.get('ENABLE_TRADING', 'true').lower() == 'true'
ENABLE_LOGGING = os.environ.get('ENABLE_LOGGING', 'true').lower() == 'true'

print(Fore.CYAN + "="*70)
print(Fore.CYAN + f"🌐 WebSocket URL: {PUBLIC_WS_URL}")
print(Fore.CYAN + "="*70)

# ✅ CORRECT CRASH AND BOOM SYMBOLS
SYMBOL_CONFIGS = {
    "CRASH500": {"symbol": "CRASH500", "type": "CRASH", "multiplier": 400},
    "CRASH600": {"symbol": "CRASH600", "type": "CRASH", "multiplier": 400},
    "CRASH900": {"symbol": "CRASH900", "type": "CRASH", "multiplier": 400},
    "CRASH1000": {"symbol": "CRASH1000", "type": "CRASH", "multiplier": 500},
    "BOOM500": {"symbol": "BOOM500", "type": "BOOM", "multiplier": 400},
    "BOOM600": {"symbol": "BOOM600", "type": "BOOM", "multiplier": 400},
    "BOOM900": {"symbol": "BOOM900", "type": "BOOM", "multiplier": 400}
}

# --- Account Functions ---
def get_demo_account_id():
    """Get DEMO account ID using REST API"""
    print(Fore.YELLOW + "📡 Getting demo account ID...")
    url = "https://api.derivws.com/trading/v1/options/accounts"
    headers = {
        "Deriv-App-ID": DERIV_APP_ID,
        "Authorization": f"Bearer {DERIV_TOKEN}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error = response.json().get('error', {}).get('message', 'Unknown error')
            raise Exception(f"Failed to get accounts: {error}")
        
        data = response.json()
        accounts = data.get('data', [])
        
        # Find the demo account
        demo_account = None
        for account in accounts:
            if account.get('account_type') == 'demo':
                demo_account = account
                break
        
        if not demo_account:
            raise Exception("No demo account found! Please create a demo account first.")
        
        print(Fore.GREEN + f"✅ Demo Account ID: {demo_account['account_id']}")
        print(Fore.GREEN + f"💰 Demo Balance: {demo_account['currency']} {demo_account['balance']}")
        
        return demo_account['account_id'], demo_account.get('currency', 'USD')
        
    except Exception as e:
        print(Fore.RED + f"❌ Error getting demo account: {e}")
        return None, None

def get_otp_url(account_id):
    """Request OTP URL for authenticated WebSocket"""
    print(Fore.YELLOW + f"📡 Getting OTP URL for demo account {account_id}...")
    url = f"https://api.derivws.com/trading/v1/options/accounts/{account_id}/otp"
    headers = {
        "Deriv-App-ID": DERIV_APP_ID,
        "Authorization": f"Bearer {DERIV_TOKEN}"
    }
    
    try:
        response = requests.post(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error = response.json().get('error', {}).get('message', 'Unknown error')
            raise Exception(f"OTP request failed: {error}")
        
        data = response.json()
        
        if not data.get('data') or not data['data'].get('url'):
            raise Exception("No OTP URL found in response")
        
        ws_url = data['data']['url']
        print(Fore.GREEN + f"✅ OTP URL obtained for demo account")
        return ws_url
    except Exception as e:
        print(Fore.RED + f"❌ Error getting OTP URL: {e}")
        return None

async def get_balance_via_otp(ws_url):
    """Get balance using OTP-authenticated WebSocket"""
    try:
        async with websockets.connect(ws_url) as websocket:
            request = {
                "balance": 1,
                "req_id": 1
            }
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            data = json.loads(response)
            
            if 'balance' in data:
                balance = data['balance'].get('balance', 0)
                currency = data['balance'].get('currency', 'USD')
                return balance, currency
            elif 'error' in data:
                print(Fore.RED + f"❌ API Error: {data['error']['message']}")
                return None, None
            else:
                print(Fore.RED + f"❌ Unexpected response: {data}")
                return None, None
    except Exception as e:
        print(Fore.RED + f"❌ WebSocket error: {e}")
        return None, None

def validate_credentials():
    """Validate credentials and get demo account balance"""
    print(Fore.CYAN + "\n🔍 Validating Deriv Credentials...")
    print(Fore.CYAN + "="*60)
    
    try:
        # Get demo account
        account_id, currency = get_demo_account_id()
        if not account_id:
            raise Exception("No demo account found!")
        
        # Get OTP URL
        otp_ws_url = get_otp_url(account_id)
        if not otp_ws_url:
            raise Exception("Failed to get OTP URL!")
        
        # Get balance
        print(Fore.YELLOW + "📡 Fetching balance via OTP...")
        balance, currency = asyncio.run(get_balance_via_otp(otp_ws_url))
        
        if balance is not None:
            print(Fore.GREEN + "="*60)
            print(Fore.GREEN + f"✅ SUCCESS! Connected to DEMO account: {account_id}")
            print(Fore.GREEN + f"💰 Balance: {currency} {balance:.2f}")
            print(Fore.GREEN + f"✅ App ID: {DERIV_APP_ID} is valid")
            print(Fore.GREEN + f"✅ Token is valid")
            print(Fore.GREEN + "="*60)
            return True, account_id, balance, currency
        else:
            raise Exception("Failed to fetch balance!")
            
    except Exception as e:
        print(Fore.RED + f"❌ Validation failed: {e}")
        print(Fore.RED + "="*60)
        return False, None, 0, "USD"

# --- Trading Bot Class ---
class CrashBoomTrader:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.active_positions = {}
        self.account_id = None
        self.balance = 0
        self.currency = "USD"
        
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.running = True
        self.websocket = None
        self.last_data = {}
        self.last_update = {}
        self.models_loaded = False
        self.recent_trades = []  # Store recent trades for display
        self.trade_log = []  # Full trade history
        
        # Validate credentials and get demo account
        is_valid, account_id, balance, currency = validate_credentials()
        if not is_valid:
            raise Exception("❌ Credential validation failed. Please check your App ID and Token.")
        
        self.account_id = account_id
        self.balance = balance
        self.currency = currency
        
        print(Fore.CYAN + "="*70)
        print(Fore.CYAN + "🔴 DERIV CRASH/BOOM TRADER - PRODUCTION (DEMO ACCOUNT)")
        print(Fore.CYAN + "="*70)
        print(Fore.GREEN + f"📡 Account ID: {self.account_id}")
        print(Fore.GREEN + f"💰 Balance: {self.currency} {self.balance:.2f}")
        print(Fore.GREEN + f"📡 App ID: {DERIV_APP_ID}")
        print(Fore.CYAN + "="*70)
        
        self.load_models()
        
        print(Fore.YELLOW + f"\n🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
        print(Fore.YELLOW + f"💲 Base Stake: ${BASE_STAKE}")
        print(Fore.YELLOW + f"📈 Take Profit: ${BASE_STAKE * TP_MULTIPLIER:.2f}")
        print(Fore.YELLOW + f"📉 Stop Loss: ${BASE_STAKE * SL_MULTIPLIER:.2f}")
        print(Fore.YELLOW + f"📊 Monitoring: {', '.join(SYMBOL_CONFIGS.keys())}")
        print(Fore.YELLOW + f"🚀 Trading: {'ENABLED' if ENABLE_TRADING else 'DISABLED'}")
        print(Fore.CYAN + "="*70)
    
    def load_models(self):
        """Load all saved ML models from their respective folders"""
        print(Fore.YELLOW + "\n📂 Loading models...")
        base_model_dir = "models"
        
        if not os.path.exists(base_model_dir):
            raise Exception(f"❌ Model directory '{base_model_dir}' not found! Models are REQUIRED for trading.")
        
        missing_models = []
        
        for symbol_name in SYMBOL_CONFIGS.keys():
            try:
                model_dir = os.path.join(base_model_dir, symbol_name)
                
                if not os.path.exists(model_dir):
                    missing_models.append(f"{symbol_name} (folder not found)")
                    print(Fore.RED + f"  ❌ {symbol_name} folder not found at: {model_dir}")
                    continue
                
                model_path = os.path.join(model_dir, 'model_60pct.pkl')
                scaler_path = os.path.join(model_dir, 'scaler_60pct.pkl')
                imputer_path = os.path.join(model_dir, 'imputer_60pct.pkl')
                
                if not os.path.exists(model_path):
                    missing_models.append(f"{symbol_name} (model_60pct.pkl missing)")
                    print(Fore.RED + f"  ❌ {symbol_name} model_60pct.pkl not found")
                    continue
                    
                if not os.path.exists(scaler_path):
                    missing_models.append(f"{symbol_name} (scaler_60pct.pkl missing)")
                    print(Fore.RED + f"  ❌ {symbol_name} scaler_60pct.pkl not found")
                    continue
                    
                if not os.path.exists(imputer_path):
                    missing_models.append(f"{symbol_name} (imputer_60pct.pkl missing)")
                    print(Fore.RED + f"  ❌ {symbol_name} imputer_60pct.pkl not found")
                    continue
                
                self.models[symbol_name] = joblib.load(model_path)
                self.scalers[symbol_name] = joblib.load(scaler_path)
                self.imputers[symbol_name] = joblib.load(imputer_path)
                print(Fore.GREEN + f"  ✅ {symbol_name} models loaded from {model_dir}")
                
            except Exception as e:
                missing_models.append(f"{symbol_name} (error: {str(e)})")
                print(Fore.RED + f"  ❌ {symbol_name} failed to load: {e}")
        
        if missing_models:
            error_msg = f"❌ Missing models for: {', '.join(missing_models)}. All models are REQUIRED for trading."
            raise Exception(error_msg)
        
        self.models_loaded = True
        print(Fore.GREEN + f"\n✅ All {len(self.models)} models loaded successfully!")
    
    async def connect_websocket(self):
        """Connect to Deriv Public WebSocket for market data"""
        try:
            if self.websocket:
                try:
                    await self.websocket.close()
                except:
                    pass
            
            self.websocket = await websockets.connect(PUBLIC_WS_URL)
            if ENABLE_LOGGING:
                print(Fore.GREEN + f"✅ Public WebSocket connected to Deriv")
            return True
        except Exception as e:
            print(Fore.RED + f"❌ Connection error: {e}")
            return False
    
    async def fetch_1min_candles(self, symbol, count=200):
        """Fetch 1-minute candle data for a symbol"""
        try:
            if not self.websocket:
                if not await self.connect_websocket():
                    return None
            
            request = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": 60,
                "count": count,
                "end": "latest"
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if 'error' in data:
                if ENABLE_LOGGING:
                    print(Fore.RED + f"❌ Error fetching {symbol}: {data['error']['message']}")
                return None
            
            if 'candles' in data and data['candles']:
                df = pd.DataFrame(data['candles'])
                df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                required_cols = ['open', 'high', 'low', 'close']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = df['close']
                
                return df[required_cols]
            
            return None
            
        except (websockets.exceptions.ConnectionClosed, 
                websockets.exceptions.WebSocketException) as e:
            if ENABLE_LOGGING:
                print(Fore.YELLOW + f"🔄 WebSocket reconnecting...")
            self.websocket = None
            return None
        except Exception as e:
            if ENABLE_LOGGING:
                print(Fore.RED + f"❌ WebSocket error for {symbol}: {e}")
            return None
    
    async def place_trade(self, symbol, contract_type):
        """Place a trade using the demo account"""
        try:
            # Check if trading is enabled
            if not ENABLE_TRADING:
                print(Fore.YELLOW + "⚠️ Trading is disabled. Set ENABLE_TRADING=true to enable.")
                return {'success': False, 'error': 'Trading disabled'}
            
            # Get fresh OTP URL for trading
            otp_ws_url = get_otp_url(self.account_id)
            if not otp_ws_url:
                print(Fore.RED + f"❌ Failed to get OTP URL for trade")
                return {'success': False, 'error': 'No OTP URL'}
            
            config = SYMBOL_CONFIGS[symbol]
            multiplier = config['multiplier']
            
            async with websockets.connect(otp_ws_url) as ws:
                # Get proposal
                proposal_request = {
                    "proposal": 1,
                    "amount": BASE_STAKE,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": self.currency,
                    "underlying_symbol": symbol,
                    "multiplier": multiplier,
                    "limit_order": {
                        "stop_loss": SL_MULTIPLIER,
                        "take_profit": TP_MULTIPLIER
                    }
                }
                
                await ws.send(json.dumps(proposal_request))
                proposal_response = await ws.recv()
                proposal_data = json.loads(proposal_response)
                
                if 'error' in proposal_data:
                    error_msg = proposal_data['error']['message']
                    print(Fore.RED + f"❌ Proposal error: {error_msg}")
                    return {'success': False, 'error': error_msg}
                
                proposal = proposal_data.get('proposal', {})
                proposal_id = proposal.get('id')
                ask_price = proposal.get('ask_price')
                
                if not proposal_id:
                    print(Fore.RED + "❌ No proposal ID received")
                    return {'success': False, 'error': 'No proposal ID'}
                
                # Buy the contract
                buy_request = {
                    "buy": proposal_id,
                    "price": ask_price
                }
                
                await ws.send(json.dumps(buy_request))
                buy_response = await ws.recv()
                buy_data = json.loads(buy_response)
                
                if 'error' in buy_data:
                    error_msg = buy_data['error']['message']
                    print(Fore.RED + f"❌ Buy error: {error_msg}")
                    return {'success': False, 'error': error_msg}
                
                buy = buy_data.get('buy', {})
                contract_id = buy.get('contract_id')
                balance_after = buy.get('balance_after')
                
                return {
                    'success': True,
                    'contract_id': contract_id,
                    'balance_after': balance_after,
                    'ask_price': ask_price,
                    'multiplier': multiplier
                }
                
        except Exception as e:
            print(Fore.RED + f"❌ Trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_features(self, df):
        """Calculate technical indicators for 1-minute data"""
        close_prices = df['close'].values.tolist()
        
        # DPO
        try:
            dpo = DPO(250, input_values=close_prices)
            df['dpo'] = dpo
        except:
            df['dpo'] = 0
        
        # MACD Histogram
        try:
            macd = MACD(36, 120, 36, input_values=close_prices)
            df['macd_hist'] = [v.histogram if v else 0 for v in macd]
        except:
            df['macd_hist'] = 0
        
        # StochRSI
        try:
            stoch_rsi = StochRSI(300, 250, 80, 9, input_values=close_prices)
            df['stoch_rsi'] = [v.k if v else 50 for v in stoch_rsi]
        except:
            df['stoch_rsi'] = 50
        
        # RSI
        try:
            rsi = RSI(14, input_values=close_prices)
            df['rsi'] = rsi
        except:
            df['rsi'] = 50
        
        # Moving Averages
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['trend'] = df['ma_20'] - df['ma_50']
        
        # Volatility
        df['range'] = df['high'] - df['low']
        df['volatility'] = df['range'] / (df['close'] + 0.0001)
        
        # Momentum
        df['momentum_pct'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 0.0001)
        
        # Price Position
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 0.0001)
        
        # Bullish/Bearish indicators
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['bullish_size_ratio'] = abs(df['close'] - df['open']) / (abs(df['close'] - df['open']).rolling(5).mean() + 0.0001)
        df['bearish_size_ratio'] = abs(df['close'] - df['open']) / (abs(df['close'] - df['open']).rolling(5).mean() + 0.0001)
        
        # Body size
        df['body'] = abs(df['close'] - df['open'])
        
        # Acceleration
        df['acceleration'] = df['momentum_pct'] - df['momentum_pct'].shift(1)
        
        # Distance from MA50
        df['distance_from_ma50'] = (df['close'] - df['ma_50']) / (df['ma_50'] + 0.0001)
        
        # Support/Resistance
        df['near_resistance'] = ((df['close'] / df['high'].rolling(20).max() - 1) > -0.005).astype(int)
        df['near_support'] = ((df['close'] / df['low'].rolling(20).min() - 1) < 0.005).astype(int)
        df['breakout_high'] = (df['high'] > df['high'].rolling(10).max().shift(1)).astype(int)
        df['breakout_low'] = (df['low'] < df['low'].rolling(10).min().shift(1)).astype(int)
        
        # Fill NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.fillna(0)
        
        return df
    
    def predict_signal(self, symbol_name, df):
        """Predict trading signal using ML model"""
        if symbol_name not in self.models:
            raise Exception(f"❌ No model found for {symbol_name}")
        
        if not self.models_loaded:
            raise Exception("❌ Models not loaded. Cannot predict.")
        
        try:
            df = self.calculate_features(df)
            latest = df.iloc[-1:].copy()
            
            # Different features for CRASH and BOOM
            if "CRASH" in symbol_name:
                feature_cols = [
                    'stoch_rsi', 'macd_hist', 'trend', 'dpo', 'rsi',
                    'momentum_pct', 'price_position', 'bearish_size_ratio',
                    'volatility', 'body', 'acceleration', 'distance_from_ma50',
                    'near_resistance', 'near_support', 'breakout_low'
                ]
            else:
                feature_cols = [
                    'stoch_rsi', 'macd_hist', 'trend', 'dpo', 'rsi',
                    'momentum_pct', 'price_position', 'bullish_size_ratio',
                    'volatility', 'body', 'acceleration', 'distance_from_ma50',
                    'near_resistance', 'near_support', 'breakout_high'
                ]
            
            # Prepare features
            feature_data = {}
            for col in feature_cols:
                if col in latest.columns:
                    feature_data[col] = latest[col].values[0]
                else:
                    feature_data[col] = 0
            
            X_signal = pd.DataFrame([feature_data]).fillna(0)
            
            # Transform using loaded models
            X_imp = pd.DataFrame(self.imputers[symbol_name].transform(X_signal), columns=X_signal.columns)
            X_scaled = self.scalers[symbol_name].transform(X_imp)
            
            # Predict
            prob = self.models[symbol_name].predict_proba(X_scaled)[0, 1]
            dpo_value = latest.get('dpo', 0).values[0] if 'dpo' in latest else 0
            
            return {
                'confidence': prob,
                'dpo': dpo_value,
                'timestamp': latest.index[0]
            }
            
        except Exception as e:
            print(Fore.RED + f"❌ Prediction error for {symbol_name}: {e}")
            return None
    
    def check_for_signal(self, symbol_name, df):
        """Check for entry signal"""
        if symbol_name in self.active_positions:
            return None
        
        config = SYMBOL_CONFIGS[symbol_name]
        symbol_type = config["type"]
        
        pred = self.predict_signal(symbol_name, df)
        if pred is None:
            return None
        
        if symbol_type == "CRASH":
            is_valid = pred['dpo'] < 0 and pred['confidence'] >= CONFIDENCE_THRESHOLD
            signal = 'SELL' if is_valid else None
        else:
            is_valid = pred['dpo'] > 0 and pred['confidence'] >= CONFIDENCE_THRESHOLD
            signal = 'BUY' if is_valid else None
        
        if not signal:
            return None
        
        entry_price = df['close'].iloc[-1]
        multiplier = config['multiplier']
        
        if symbol_type == "CRASH":
            tp_price = entry_price - (BASE_STAKE * TP_MULTIPLIER / multiplier)
            sl_price = entry_price + (BASE_STAKE * SL_MULTIPLIER / multiplier)
        else:
            tp_price = entry_price + (BASE_STAKE * TP_MULTIPLIER / multiplier)
            sl_price = entry_price - (BASE_STAKE * SL_MULTIPLIER / multiplier)
        
        return {
            'symbol': symbol_name,
            'signal': signal,
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'confidence': pred['confidence'],
            'dpo': pred['dpo'],
            'multiplier': multiplier,
            'timestamp': pred['timestamp']
        }
    
    def manage_position(self, symbol_name, df):
        """Manage open position - check TP/SL"""
        if symbol_name not in self.active_positions:
            return
        
        position = self.active_positions[symbol_name]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        config = SYMBOL_CONFIGS[symbol_name]
        symbol_type = config["type"]
        
        if symbol_type == "CRASH":
            if current_low <= position['tp_price']:
                profit = BASE_STAKE * TP_MULTIPLIER
                self.close_position(symbol_name, 'TAKE PROFIT', profit)
                return
            if current_high >= position['sl_price']:
                profit = -BASE_STAKE * SL_MULTIPLIER
                self.close_position(symbol_name, 'STOP LOSS', profit)
                return
        else:
            if current_high >= position['tp_price']:
                profit = BASE_STAKE * TP_MULTIPLIER
                self.close_position(symbol_name, 'TAKE PROFIT', profit)
                return
            if current_low <= position['sl_price']:
                profit = -BASE_STAKE * SL_MULTIPLIER
                self.close_position(symbol_name, 'STOP LOSS', profit)
                return
    
    def close_position(self, symbol_name, reason, profit):
        """Close position and update statistics"""
        position = self.active_positions.pop(symbol_name)
        
        self.trade_count += 1
        if profit > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.total_profit += profit
        self.balance += profit
        
        # Store in recent trades
        trade_record = {
            'symbol': symbol_name,
            'reason': reason,
            'profit': profit,
            'confidence': position['confidence'],
            'time': datetime.now().strftime('%H:%M:%S'),
            'contract_id': position.get('contract_id', 'N/A')
        }
        self.recent_trades.insert(0, trade_record)
        self.trade_log.append(trade_record)
        if len(self.recent_trades) > 20:
            self.recent_trades = self.recent_trades[:20]
        if len(self.trade_log) > 1000:
            self.trade_log = self.trade_log[-1000:]
        
        emoji = "✅" if profit > 0 else "❌"
        color = Fore.GREEN if profit > 0 else Fore.RED
        
        print(color + f"\n{emoji} POSITION CLOSED - {symbol_name}")
        print(color + f"   {reason}: ${profit:.2f}")
        print(color + f"   Confidence: {position['confidence']:.2%}")
        print(color + f"   Trade #{self.trade_count}")
        win_rate = self.win_count/self.trade_count*100 if self.trade_count > 0 else 0
        print(color + f"   Win Rate: {win_rate:.1f}%")
        print(color + f"   Balance: ${self.balance:.2f}")
    
    def execute_trade(self, signal):
        """Execute a new trade"""
        print(Fore.CYAN + f"\n🟢 TRADE SIGNAL DETECTED - {signal['symbol']}")
        print(Fore.CYAN + f"   Signal: {signal['signal']}")
        print(Fore.CYAN + f"   Confidence: {signal['confidence']:.2%}")
        print(Fore.CYAN + f"   DPO: {signal['dpo']:.4f}")
        print(Fore.CYAN + f"   Entry Price: {signal['entry_price']:.4f}")
        print(Fore.CYAN + f"   TP: {signal['tp_price']:.4f} (${BASE_STAKE * TP_MULTIPLIER:.2f})")
        print(Fore.CYAN + f"   SL: {signal['sl_price']:.4f} (-${BASE_STAKE * SL_MULTIPLIER:.2f})")
        
        # Determine contract type
        contract_type = "MULTUP" if signal['signal'] == "BUY" else "MULTDOWN"
        
        # Place the trade
        result = asyncio.run(self.place_trade(signal['symbol'], contract_type))
        
        if result and result.get('success'):
            self.active_positions[signal['symbol']] = {
                'entry_time': signal['timestamp'],
                'entry_price': signal['entry_price'],
                'tp_price': signal['tp_price'],
                'sl_price': signal['sl_price'],
                'confidence': signal['confidence'],
                'multiplier': signal['multiplier'],
                'contract_id': result['contract_id']
            }
            
            print(Fore.GREEN + f"\n✅ TRADE EXECUTED - {signal['symbol']}")
            print(Fore.GREEN + f"   Contract ID: {result['contract_id']}")
            print(Fore.GREEN + f"   Balance After: ${result['balance_after']:.2f}")
            self.balance = result['balance_after']
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Unknown error'
            print(Fore.RED + f"\n❌ Trade failed for {signal['symbol']}: {error_msg}")
    
    async def run_trading_loop(self):
        """Main trading loop - runs every minute - NO RATE LIMITING"""
        print(Fore.CYAN + "\n🟢 STARTING LIVE TRADING LOOP")
        print(Fore.CYAN + "Fetching 1-minute candles for all symbols every 60 seconds...")
        print(Fore.YELLOW + f"⚡ Trading on EVERY valid signal (no rate limiting)")
        print(Fore.YELLOW + f"🚀 Trading: {'ENABLED' if ENABLE_TRADING else 'DISABLED'}")
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                print(Fore.CYAN + f"\n⏰ Cycle #{cycle_count} - {current_time}")
                
                # Fetch and analyze all symbols
                for symbol_name, config in SYMBOL_CONFIGS.items():
                    df = await self.fetch_1min_candles(config['symbol'], count=200)
                    
                    if df is None or df.empty:
                        if ENABLE_LOGGING:
                            print(Fore.YELLOW + f"  ⚠️ No data for {symbol_name}")
                        continue
                    
                    self.last_data[symbol_name] = df
                    self.last_update[symbol_name] = datetime.now()
                    
                    # Show prediction confidence
                    pred = self.predict_signal(symbol_name, df)
                    if pred and ENABLE_LOGGING:
                        conf_str = f"Conf: {pred['confidence']:.1%}" if pred['confidence'] else "N/A"
                        dpo_str = f"DPO: {pred['dpo']:.4f}" if pred['dpo'] else "N/A"
                        print(Fore.WHITE + f"  📊 {symbol_name}: close: {df['close'].iloc[-1]:.4f} | {conf_str} | {dpo_str}")
                    elif ENABLE_LOGGING:
                        print(Fore.WHITE + f"  📊 {symbol_name}: close: {df['close'].iloc[-1]:.4f}")
                    
                    # Manage existing position
                    self.manage_position(symbol_name, df)
                    
                    # Check for new signal (only if not in position)
                    if symbol_name not in self.active_positions:
                        signal = self.check_for_signal(symbol_name, df)
                        if signal:
                            self.execute_trade(signal)
                
                # Print status summary
                win_rate = self.win_count/self.trade_count*100 if self.trade_count > 0 else 0
                print(Fore.CYAN + f"\n📊 STATUS SUMMARY:")
                print(Fore.CYAN + f"   Trades: {self.trade_count} (Wins: {Fore.GREEN}{self.win_count}{Fore.CYAN} | Losses: {Fore.RED}{self.loss_count}{Fore.CYAN})")
                print(Fore.CYAN + f"   Win Rate: {win_rate:.1f}%")
                print(Fore.CYAN + f"   Total P&L: {Fore.GREEN if self.total_profit >= 0 else Fore.RED}${self.total_profit:.2f}")
                print(Fore.CYAN + f"   Balance: ${self.balance:.2f}")
                print(Fore.CYAN + f"   Active Positions: {len(self.active_positions)}")
                
                # Wait 60 seconds before next cycle
                await asyncio.sleep(60)
                
            except Exception as e:
                print(Fore.RED + f"❌ Error in trading loop: {e}")
                await asyncio.sleep(60)
        
        if self.websocket:
            await self.websocket.close()
        print(Fore.CYAN + "\n🛑 Trading loop stopped")

# --- Flask Routes for Render ---
@app.route('/')
def home():
    if 'trader' not in globals():
        return jsonify({"error": "Trader not initialized"}), 500
    
    status = {
        "status": "running",
        "bot": "Deriv Crash/Boom Trader (DEMO)",
        "account_id": trader.account_id if trader else None,
        "balance": f"{trader.currency} {trader.balance:.2f}" if trader else "USD 0.00",
        "app_id": DERIV_APP_ID,
        "symbols": list(SYMBOL_CONFIGS.keys()),
        "models_loaded": trader.models_loaded if trader else False,
        "trade_count": trader.trade_count if trader else 0,
        "win_rate": f"{trader.win_count/trader.trade_count*100:.1f}%" if trader and trader.trade_count > 0 else "0%",
        "total_profit": f"${trader.total_profit:.2f}" if trader else "$0.00",
        "active_positions": len(trader.active_positions) if trader else 0,
        "trading_enabled": ENABLE_TRADING,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(status)

@app.route('/status')
def status():
    if 'trader' not in globals():
        return jsonify({"error": "Trader not initialized"}), 500
    
    active_positions = {}
    for symbol, pos in trader.active_positions.items():
        active_positions[symbol] = {
            "entry_price": pos['entry_price'],
            "tp_price": pos['tp_price'],
            "sl_price": pos['sl_price'],
            "confidence": pos['confidence'],
            "multiplier": pos['multiplier'],
            "contract_id": pos.get('contract_id', 'N/A')
        }
    
    return jsonify({
        "status": "running",
        "account_id": trader.account_id,
        "balance": f"{trader.currency} {trader.balance:.2f}",
        "app_id": DERIV_APP_ID,
        "models_loaded": trader.models_loaded,
        "trade_count": trader.trade_count,
        "win_count": trader.win_count,
        "loss_count": trader.loss_count,
        "win_rate": f"{trader.win_count/trader.trade_count*100:.1f}%" if trader.trade_count > 0 else "0%",
        "total_profit": trader.total_profit,
        "active_positions": active_positions,
        "recent_trades": trader.recent_trades[:10] if hasattr(trader, 'recent_trades') else [],
        "trading_enabled": ENABLE_TRADING,
        "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/toggle_trading')
def toggle_trading():
    """Toggle trading on/off"""
    global ENABLE_TRADING
    ENABLE_TRADING = not ENABLE_TRADING
    status = "ENABLED" if ENABLE_TRADING else "DISABLED"
    print(Fore.YELLOW + f"🔄 Trading {status}")
    return jsonify({"trading_enabled": ENABLE_TRADING, "message": f"Trading {status}"})

@app.route('/stop')
def stop_bot():
    if 'trader' in globals():
        trader.running = False
        return jsonify({"message": "Bot stopping..."})
    return jsonify({"error": "Trader not found"}), 404

def run_trading_bot():
    global trader
    try:
        trader = CrashBoomTrader()
        asyncio.run(trader.run_trading_loop())
    except Exception as e:
        print(Fore.RED + f"❌ Fatal error in trading bot: {e}")
        raise

if __name__ == "__main__":
    if not DERIV_APP_ID:
        print(Fore.RED + "❌ ERROR: DERIV_APP_ID is REQUIRED!")
        exit(1)
    
    if not DERIV_TOKEN:
        print(Fore.RED + "❌ ERROR: DERIV_TOKEN is REQUIRED!")
        exit(1)
    
    try:
        trader = CrashBoomTrader()
        trading_thread = threading.Thread(target=run_trading_bot, daemon=True)
        trading_thread.start()
    except Exception as e:
        print(Fore.RED + f"❌ Failed to initialize trading bot: {e}")
        exit(1)
    
    port = int(os.environ.get('PORT', 10000))
    print(Fore.GREEN + f"\n🚀 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)