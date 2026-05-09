import re
import math
import chess
import chess.engine
from typing import List, Dict, Union

# --- Configuration Constants ---
STOCKFISH_PATH = "stockfish/stockfish-ubuntu-x86-64-avx2" 
STOCKFISH_DEPTH = 15
ENGINE_TIME_LIMIT = 0.1

FORMAT_FAIL_REWARD = 0.0
INVALID_MOVE_NOTATION_REWARD = 0.01
ILLEGAL_MOVE_REWARD = 0.1
REWARD_SCALING_FACTOR = 200.0

# Initialize the synchronous Stockfish engine globally
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def _get_move_eval(fen: str, move_str: str) -> float:
    """Evaluate a move in centipawns using synchronous Stockfish."""
    board = chess.Board(fen)
    try:
        # Try UCI first since the dataset provides UCI moves
        move = board.parse_uci(move_str)
    except ValueError:
        # Fallback to SAN if model outputs SAN
        move = board.parse_san(move_str)
        
    board.push(move)
    info = engine.analyse(
        board, 
        chess.engine.Limit(depth=STOCKFISH_DEPTH, time=ENGINE_TIME_LIMIT)
    )
    # perspective=not board.turn because the move was just made by the previous player
    score = info["score"].pov(not board.turn).score(mate_score=10000)
    return float(score)

def stockfish_reward(
    prompts: List[Union[str, List[Dict[str, str]]]], 
    completions: List[Union[str, List[Dict[str, str]]]], 
    **kwargs
) -> List[float]:
    """
    TRL-compatible reward function.
    Uses 'fen' and 'uci_moves' (where uci_moves[0] is the best move).
    """
    fens = kwargs.get("fen", [])
    # uci_moves is expected to be a list of lists (batch size x move list)
    uci_moves_batch = kwargs.get("uci_moves", [])
    
    if not fens or not uci_moves_batch:
        raise ValueError("Dataset must contain 'fen' and 'uci_moves' columns.")

    rewards = []
    best_move_eval_cache = {}

    for completion, fen, moves_list in zip(completions, fens, uci_moves_batch):
        # 1. Extract the best move (the first element in the list)
        if not moves_list or len(moves_list) == 0:
            rewards.append(0.0) # Safety check for empty lists
            continue
            
        target_best_move = moves_list[0]

        # 2. Extract model response text
        if isinstance(completion, list):
            content = completion[-1]["content"]
        else:
            content = completion

        # 3. Format Validation (<think> and <answer> tags)
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)

        if not think_match or not answer_match:
            rewards.append(FORMAT_FAIL_REWARD)
            continue

        model_move_str = answer_match.group(1).strip()
        board = chess.Board(fen)

        # 4. Notation Legality (SAN or UCI)
        try:
            # We try UCI first as it's more standard for LLM chess, then SAN
            try:
                move = board.parse_uci(model_move_str)
            except ValueError:
                move = board.parse_san(model_move_str)
        except ValueError:
            rewards.append(INVALID_MOVE_NOTATION_REWARD)
            continue

        # 5. Rule Legality
        if move not in board.legal_moves:
            rewards.append(ILLEGAL_MOVE_REWARD)
            continue

        # 6. Evaluation Logic
        # Cache key includes the FEN and the move to avoid redundant engine calls
        cache_key = f"{fen}_{target_best_move}"
        if cache_key not in best_move_eval_cache:
            best_move_eval_cache[cache_key] = _get_move_eval(fen, target_best_move)
        
        eval_best = best_move_eval_cache[cache_key]
        eval_chosen = _get_move_eval(fen, model_move_str)

        # 7. Scaled Reward Calculation
        # Reward = 1 / (1 + exp(diff / scale))
        eval_diff = max(0.0, eval_best - eval_chosen)
        final_score = 1.0 / (1.0 + math.exp(eval_diff / REWARD_SCALING_FACTOR))
        
        rewards.append(final_score)

    return rewards