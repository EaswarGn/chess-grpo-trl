import json
import requests
from transformers import AutoTokenizer

# 1. Define the server URL and the model name to initialize the tokenizer
# Replace with your actual server IP/port and Hugging Face model identifier
SERVER_URL = "http://0.0.0.0:8000/chat/"
MODEL_NAME = "codingmonster1234/Llama-3.1-8B-Instruct-Chess-Reasoning-SFT-v2"  # Change to the model running on the server

# Initialize the tokenizer to decode token IDs back into English text
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Construct the payload matching the /chat/ schema of vllm_serve.py
payload = {
    "messages": [
        [
            {
                "content": "You are an elite chess AI, functioning at a Grandmaster level of tactical and positional understanding. Your task is to analyze a given chess position, accurately calculate variations, and determine the absolute best move. \n\nYou must think deeply and systematically before providing your final answer. Your internal reasoning must be enclosed within <think> and </think> tags. After completing your analysis, you must output your final chosen move enclosed within <answer> and </answer> tags.\n\nYour reasoning trace within the <think> tags must follow a rigorous, structured analytical process consisting of the following sequential steps:\n\n## Step 1: FEN parsing\nDeconstruct the provided board state rank by rank. Accurately identify the exact square of every piece on the board. Do not hallucinate piece placements. \n\n## Step 2: Piece Positions\nCategorize the active pieces for both White and Black. Note the exact coordinate of the kings, heavy pieces (queens, rooks), minor pieces, and relevant pawn structures.\n\n## Step 3: Position Summary\nProvide a high-level, critical evaluation of the position. Assess material imbalances, king safety, pawn structure (e.g., passed pawns, weaknesses), piece activity, and critical tensions (pins, skewers, overloaded defenders). Identify whose turn it is to move.\n\n## Step 4: Candidate Moves\nGenerate a comprehensive list of candidate moves. You must actively look for forcing moves: Checks, Captures, and Threats (CCT). Broaden your search to include prophylactic moves or positional improvements if no immediate tactical sequence exists. List the raw candidate moves clearly.\n\n## Step 5: Candidate Lines Analysis\nCalculate concrete variations for each candidate move. \n- Visualize the board after each ply. \n- Always calculate your opponent's most critical and resilient replies, not just cooperative moves.\n- Evaluate the resulting terminal position of each calculated line. \n- Look for tactical motifs: back-rank weaknesses, discovered attacks, deflections, and removal of the defender.\n- Compare the outcomes of the candidate lines to determine which move forces a tangible advantage (mate, material gain, or decisive positional superiority) or maintains equality in a worse position.\n\n## Step 6: Solution\nFormat your final conclusion using standard algebraic notation. Include brief, precise annotations (using `{}` for comments and standard chess punctuation like `!`, `?`, `+`, `#`) detailing the primary forcing line and why alternative lines fail.\n\nOnce your reasoning is complete, output only the single best move in standard algebraic notation inside the <answer> tags.\n\nExample Output Format:\n<think>\n[Detailed reasoning following the 6 steps above]\n</think>\n<answer>Move</answer>\n",
                "role": "system"
            },
            {
                "content": "FEN String of current position: 2k5/1p6/p1r5/3pR3/6P1/3p4/PPP5/2KR2r1 b - - 3 32\nHere is the ASCII representation of the board:\n. . k . . . . .\n. p . . . . . .\np . r . . . . .\n. . . p R . . .\n. . . . . . P .\n. . . p . . . .\nP P P . . . . .\n. . K R . . r .\nList of legal moves in SAN format: Kd8, Kb8, Kd7, Kc7, Rc7, Rh6, Rg6, Rf6, Re6, Rd6, Rb6, Rc5, Rc4, Rc3, Rxc2+, Rxg4, Rg3, Rg2, Rh1, Rf1, Re1, Rxd1+, dxc2, b6, a5, d4, d2+, b5\nBlack to play, find the best move.\n",
                "role": "user"
            }
        ]
    ],
    "max_tokens": 2048,  # Expanded since reasoning traces require a longer generation window
    "temperature": 0.0,
}

# 3. Send the POST request to the server
print("Sending request to server...")
response = requests.post(SERVER_URL, json=payload)

# 4. Process and decode the response
if response.status_code == 200:
    response_data = response.json()
    
    # Extract completion_ids from the response structure
    # According to ChatResponse, completion_ids is a list[list[int]]
    completion_ids = response_data.get("completion_ids", [])
    
    if completion_ids:
        # Decode the first sequence generated back into English text
        decoded_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        print("\n--- Model Response ---")
        print(decoded_text)
    else:
        print("Error: No completion IDs found in response.")
else:
    print(f"Server Error ({response.status_code}): {response.text}")