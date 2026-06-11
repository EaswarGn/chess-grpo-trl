import re
import math
import chess
import chess.engine
from typing import List, Dict, Union

# --- Configuration Constants ---
STOCKFISH_PATH = "stockfish/stockfish-ubuntu-x86-64-avx2" 
STOCKFISH_DEPTH = 12  # Dropped slightly to 12 for massive step-speedups (negligible accuracy loss for puzzles)
ENGINE_TIME_LIMIT = 0.05
REWARD_SCALING_FACTOR = 200.0

# --- Dense Formatting Rewards (Curriculum) ---
REW_NO_FORMAT = 0.0
REW_HAS_THINK = 0.15     # Partial credit for getting the reasoning tags down
REW_HAS_BOTH_TAGS = 0.4  # Big reward boost just for formatting successfully 
REW_INVALID_MOVE = 0.45  # Small nudge if format is right, but the text inside isn't chess notation
REW_ILLEGAL_MOVE = 0.5   # Nudge if notation is legal style, but move isn't playable on the board

# Global lazy initializer to prevent Multi-GPU/DDP process deadlocks
_GLOBAL_ENGINE = None

def get_engine():
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    return _GLOBAL_ENGINE

def _get_move_eval(fen: str, move_obj: chess.Move) -> float:
    """Evaluate a validated Move object using the process-safe engine."""
    board = chess.Board(fen)
    board.push(move_obj)
    
    sf = get_engine()
    info = sf.analyse(
        board, 
        chess.engine.Limit(depth=STOCKFISH_DEPTH, time=ENGINE_TIME_LIMIT)
    )
    score = info["score"].pov(not board.turn).score(mate_score=10000)
    return float(score)

def stockfish_reward(
    prompts: List[Union[str, List[Dict[str, str]]]], 
    completions: List[Union[str, List[Dict[str, str]]]], 
    **kwargs
) -> List[float]:
    
    fens = kwargs.get("fen", [])
    uci_moves_batch = kwargs.get("uci_moves", [])
    
    if not fens or not uci_moves_batch:
        raise ValueError("Dataset must contain 'fen' and 'uci_moves' columns.")

    rewards = []
    best_move_eval_cache = {}

    for completion, fen, moves_list in zip(completions, fens, uci_moves_batch):
        if not moves_list:
            rewards.append(0.0)
            continue
            
        target_best_move_str = moves_list[0]

        if isinstance(completion, list):
            content = completion[-1]["content"]
        else:
            content = completion

        # 1. SHAPED FORMATTING EVALUATION
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)

        # Stage 1 Failure: Complete formatting miss
        if not think_match and not answer_match:
            rewards.append(REW_NO_FORMAT)
            continue
            
        # Stage 2 Partial: Got the think tag but missed the answer tag
        if think_match and not answer_match:
            rewards.append(REW_HAS_THINK)
            continue

        # Stage 3 Partial: Got both tags open and closed
        # If it reaches here, both think_match and answer_match exist
        model_move_str = answer_match.group(1).strip()
        
        # Guard against empty answer tags: <answer></answer>
        if not model_move_str:
            rewards.append(REW_HAS_BOTH_TAGS)
            continue

        # 2. NOTATION & LEGALITY PARSING
        board = chess.Board(fen)
        move = None

        try:
            # Try UCI first
            move = board.parse_uci(model_move_str)
            if move not in board.legal_moves:
                rewards.append(REW_ILLEGAL_MOVE)
                continue
        except ValueError:
            try:
                # Fallback to SAN (Natively throws ValueError if illegal)
                move = board.parse_san(model_move_str)
            except ValueError:
                # Correct tags, but text inside isn't readable chess notation
                rewards.append(REW_INVALID_MOVE)
                continue

        # 3. CHESS EVALUATION LOGIC (Strictly >= 0.5 up to 1.0)
        try:
            try:
                target_move_obj = board.parse_uci(target_best_move_str)
            except ValueError:
                target_move_obj = board.parse_san(target_best_move_str)

            # String/Object equality short-circuit to save CPU time and yield a clean 1.0
            if move == target_move_obj:
                rewards.append(1.0)
                continue

            # Fallback to engine calculation if the model found a *different* move
            cache_key = f"{fen}_{target_best_move_str}"
            if cache_key not in best_move_eval_cache:
                best_move_eval_cache[cache_key] = _get_move_eval(fen, target_move_obj)
            
            eval_best = best_move_eval_cache[cache_key]
            eval_chosen = _get_move_eval(fen, move)

            # Exponential scaling from 0.0 to 0.5
            eval_diff = max(0.0, eval_best - eval_chosen)
            quality_score = math.exp(-eval_diff / REWARD_SCALING_FACTOR)
            
            # Map the quality score (0.0 to 1.0) into the remaining reward space (0.5 to 1.0)
            # Final Reward = 0.5 + (0.5 * quality_score)
            final_score = REW_ILLEGAL_MOVE + ((1.0 - REW_ILLEGAL_MOVE) * quality_score)
            rewards.append(final_score)
            
        except Exception:
            # Safe system fallback
            rewards.append(REW_ILLEGAL_MOVE)

    return rewards