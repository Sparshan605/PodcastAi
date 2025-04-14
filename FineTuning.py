import torch
from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Output directory for model and checkpoints
output_dir = "mistral7b-podcast"
os.makedirs(output_dir, exist_ok=True)

# Check for GPU
if not torch.cuda.is_available():
    raise SystemError("CUDA GPU is required for Unsloth 4-bit training")

# --- Unsloth imports ---
import unsloth
from unsloth import FastMistralModel
from trl import SFTTrainer

# Load model with Unsloth (4-bit)
print("Loading Mistral 7B with Unsloth...")
model, tokenizer = FastMistralModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_seq_length=2048,
    load_in_4bit=True,
)
print("Model loaded.")

# Apply LoRA
if hasattr(FastMistralModel, "get_peft_model"):
    model = FastMistralModel.get_peft_model(
        model,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
    )
else:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

# Load dataset (you can reduce split size to speed it up)
print("Loading dataset from Unsloth_data.jsonl")
dataset = load_dataset(
    "json", 
    data_files="/content/Unsloth_data.jsonl", 
    split="train[:1000]",
)

# Format dataset for instruction tuning
def format_for_unsloth(example):
    return {
        "text": f"<s>[INST] {example['prompt']} [/INST] {example['completion']}</s>"
    }

formatted_dataset = dataset.map(format_for_unsloth, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=1,  
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    logging_steps=10,
    fp16=True,
    optim="adamw_torch",
    logging_dir=f"{output_dir}/logs",
    resume_from_checkpoint=True,
)

# Trainer selection
if hasattr(unsloth, "UnslothTrainer"):
    from unsloth import UnslothTrainer
    print("Using UnslothTrainer")
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=formatted_dataset,
        max_seq_length=2048,
        formatting_func=None,
    )

# Train the model
print("Training started")
trainer.train()

# Save final model + tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model and tokenizer saved to:", output_dir)



# I am saving this file of code for future purpose if i had pricing on Groq or other llm to host i would have used this code but i am keeping 
# this for future use case to study or practise . but i did learn how to fine tune using hostin g platrfomrs.

# from groq import Groq
# import time
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("API_KEY")
# if not api_key:
#     raise Exception("API_KEY not found. Please add it to your .env file.")

# client = Groq(api_key=api_key)

# fine_tune_data = "fine_tune_data.jsonl"

# def load_file(filepath):
#     with open(filepath, "rb") as f:
#         response = client.files.create(
#             file=f,
#             purpose="fine-tune"
#         )
#     print(f"loaded file ID: {response.id}")
#     return response.id

# def start_fine_tune(file_id, base_model="llama-3.3-70b-versatile"):
#     response = client.fine_tuning.jobs.create(
#         training_file=file_id,
#         model=base_model
#     )
#     print(f"Started fine-tune job ID: {response.id}")
#     return response.id


# def monitor_job(fine_tune_id):
#     while True:
#         response = client.fine_tuning.jobs.retrieve(fine_tune_id)
#         status = response.status
#         print(f"[Status]: {status}")

#         if status == "succeeded":
#             print(f"✅ Fine-tuning complete! Your model ID is: {response.fine_tuned_model}")
#             return response.fine_tuned_model
#         elif status in ["failed", "cancelled"]:
#             print(f"❌ Fine-tuning {status}")
#             return None

#         time.sleep(60)  # check every 60 seconds

# # Main execution
# if __name__ == "__main__":
#     file_id = load_file(fine_tune_data)
#     job_id = start_fine_tune(file_id)
#     model_id = monitor_job(job_id)

# client = Groq()
# completion = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are an AI that generates engaging podcast scripts like Joe Rogan"
#         },
#         {
#             "role": "assistant",
#             "content": "Guest"
#         },
#         {
#             "role": "assistant",
#             "content": " Introduction:\n[ Intro music fades out, and I, the host, enthusiastically introduce our guest ]\nHost: \"Welcome back to another episode of 'The Uncharted Zone'! Today, we've got an incredible guest, someone who's been pushing the boundaries of human knowledge and exploration. Please give a warm welcome to the one and only, Dr. Maria Rodriguez, a renowned astrobiologist and expert on extraterrestrial life! Dr. Rodriguez, thanks for joining us, and I'm stoked to dive into some mind-blowing conversations with you!\"\n\n[ Brief pause for audience applause ]\n\nSegment 1: Introduction and Background\nHost: \"Dr. Rodriguez, for those who might not be familiar with your work, can you give us a brief background on what inspired you to become an astrobiologist, and what drives your passion for searching for life beyond Earth?\"\n\nDr. Rodriguez: \"Thanks for having me! Growing up, I was always fascinated by the night sky and the mysteries of the universe. As a child, I would spend hours gazing up at the stars, wondering if we're alone in the universe. This curiosity led me to pursue a career in astrobiology, and now, I'm dedicated to exploring the possibility of life existing elsewhere in our cosmos.\"\n\nHost: \"That's amazing! I've always been fascinated by the same questions. What do you think is the most compelling evidence for the existence of extraterrestrial life, and what are some of the most promising areas of research in this field?\"\n\nDr. Rodriguez: \"Well, there are several lines of evidence that suggest the possibility of life existing elsewhere. For example, the discovery of exoplanets, particularly those that orbit within the habitable zones of their respective stars, increases the likelihood of finding life. Additionally, the detection of biosignatures, such as the presence of oxygen or methane in a planet's atmosphere, could indicate biological activity. Currently, I'm working on a project to search for life on Mars using a novel approach that involves analyzing the planet's subsurface for signs of microbial life.\"\n\nHost: \"Whoa, that's fascinating! The idea of finding life on Mars is mind-blowing. What are some of the challenges you face in your research, and how do you overcome them?\"\n\nDr. Rodriguez: \"One of the biggest challenges is the harsh environment on Mars, which makes it difficult to design and operate equipment that can withstand the extreme conditions. Additionally, the distance between Earth and Mars creates a significant communication delay, making real-time communication with rovers and landers a challenge. To overcome these challenges, we're developing new technologies and strategies, such as using robotics and artificial intelligence to enhance our exploration capabilities.\"\n\nSegment 2: The Search for Life Beyond Earth\nHost: \"That's really interesting. Now, let's talk about some of the more speculative aspects of your research. What are your thoughts on the possibility of intelligent life existing elsewhere in the universe, and do you think we'll ever make contact with an extraterrestrial civilization?\"\n\nDr. Rodriguez: \"While we have no definitive evidence of intelligent life, I believe the probability of its existence is quite high. The universe is vast, with billions of galaxies, each containing billions of stars, and potentially, billions of planets. The emergence of intelligent life might be a rare event, but it's not impossible. As for making contact, it's difficult to predict, but I think it's essential to continue exploring and searching for signs of life, as it could have a profound impact on human society and our understanding of the universe.\"\n\nHost: \"That's a great point. What do you think is the most significant implication of finding extraterrestrial life, whether it's intelligent or not, and how do you think it would change our perspective on human existence and our place in the universe?\"\n\nDr. Rodriguez: \"The discovery of extraterrestrial life would be a paradigm-shifting moment for humanity. It would challenge our current understanding of the universe, our place within it, and our responsibilities as a species. If we were to find evidence of life elsewhere, it would raise questions about the origin of life, the possibility of a universal genetic code, and the potential for life to exist in forms that are beyond our current comprehension.\"\n\nSegment 3: The Future of Space Exploration\nHost: \"That's a great point. As we continue to explore and search for life beyond Earth, what do you think is the most crucial step we need to take as a species to ensure the long-term survival of humanity, and how can space exploration contribute to that goal?\"\n\nDr. Rodriguez: \"I believe that becoming a multi-planetary species is essential for our long-term survival. By establishing human settlements on other planets, we can ensure the continuation of our species, even if a global catastrophe were to occur on Earth. Space exploration can contribute to this goal by providing us with the necessary technologies, resources, and knowledge to establish sustainable human presence on other planets.\"\n\nHost: \"That's a great perspective. Finally, what advice would you"
#         }
#     ],
#     temperature=1,
#     max_completion_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")




