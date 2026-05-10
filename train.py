from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewards import stockfish_reward
from preprocess_dataset import process_dataset




# 0. Configuration & Hyperparameters
MODEL_NAME = "codingmonster1234/chess-sft-modelv2" 
OUTPUT_DIR = "grpo-trl-chess_env"

processed_dataset = process_dataset("codingmonster1234/chess-puzzles-rlvr", n=10000)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

training_args = GRPOConfig(
    # --- Training Loop & Progress ---
    output_dir=OUTPUT_DIR,
    run_name="grpo-chess-puzzles-v1",
    max_steps=100,               # Set to 100 steps per request
    learning_rate=5e-6,
    beta=0.04,                   # KL divergence coefficient (standard for GRPO)
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=True,                   # Efficient for A100/H100 or newer GPUs
    weight_decay=0.01,
    max_grad_norm=1.0,           # Prevents exploding gradients during RL

    # --- GRPO Specifics (Rollout) ---
    num_generations=2,           # 'G' in GRPO: completions per prompt
    max_completion_length=1024,  # Enough headroom for Chain of Thought reasoning
    temperature=0.1,             # Encourages exploration in completions
    num_generations_eval=100,
    
    # --- Batching & Throughput ---
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, # Effective Batch Size = 1 * 4 * 8 (G) = 32
    gradient_checkpointing=True,   # Saves VRAM by recomputing activations
    
    # --- Checkpointing & Validation ---
    eval_steps=20,               # Validate every 20 steps
    save_strategy="steps",
    save_steps=20,               # Checkpoint every 20 steps
    save_total_limit=3,          # Keep only the last 3 checkpoints to save disk space
    push_to_hub=False,           # Set to True if you want to sync to HF Hub
    
    # --- Observability ---
    report_to="wandb",           # Vital for monitoring reward vs. KL divergence
    log_completions=True,        # Logs generated text to W&B for inspection
    
    
)

"""# --- System & vLLM Integration ---
    # Since you're targeting shared/colocated mode for ThunderCompute/Vast:
    use_vllm=True,               
    vllm_gpu_memory_utilization=0.3, # Leave room for the Trainer process,
    vllm_mode="colocate",
    vllm_max_model_length=2048,
    vllm_enable_sleep_mode=True,"""

# 5. Initialize & Launch Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=stockfish_reward,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"]
)

trainer.train()