# ==================== SSL FIX ====================
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ==================== CORE IMPORTS ====================
import discord
from discord import app_commands
from discord.ext import commands
import torch
import re
import sys

# ==================== DEVICE ====================F
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# ==================== LOCAL MODEL IMPORTS ====================
sys.path.append("..")
from transformer.model import GPTLanguageModel
from minbpe import RegexTokenizer

# ==================== HF MODEL IMPORTS ====================
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================================================
# ==================== FORMATTING ======================
# ======================================================

# ----- Local model formatting -----
def format_local_prompt(user_input: str) -> str:
    return f"<|startoftext|>You<|separator|>{user_input}"

def format_local_output(raw_text: str) -> str:
    text = (
        raw_text.replace("<|startoftext|>", "")
        .replace("<|endoftext|>", "")
        .replace("<|separator|>", "\n")
        .replace("\\n", "\n")
    )

    text = re.sub(r'["\\\',]+', '', text)
    text = re.sub(r'\bAssistant\b', 'Tyler:', text)
    text = re.sub(r'\bYou\b', 'You:', text)
    text = re.sub(r'(?<!^)(Tyler:|You:)', r'\n\1', text)
    text = re.sub(r'\n+', '\n', text).strip()

    return text

# ----- GPT-2 formatting -----
def format_gpt2_prompt(user_input: str) -> str:
    return f"User: {user_input}\nYou:"

def clean_gpt2_output(text: str) -> str:
    if "You:" in text:
        text = text.split("You:", 1)[1]

    text = text.split("\n")[0]
    text = text.split("User:")[0]
    text = text.split("You:")[0]

    return text.strip()

# ----- Modern model formatting -----
def clean_modern_output(text: str) -> str:
    text = text.strip()

    bad_prefixes = [
        "assistant",
        "Assistant:",
        "assistant:",
        "You:",
        "Tyler:"
    ]

    for prefix in bad_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text

# ======================================================
# ==================== MODEL WRAPPERS ==================
# ======================================================

class LocalTylerModel:
    def __init__(self):
        self.tokenizer = RegexTokenizer()
        self.tokenizer.load(
            model_file="/Users/tanishsatpal/Personal Stuff/Educational/Personal Coding Projects/PycharmProjects/AI Synchronised/Tokens/Tokens v1 - Might fail.model"
        )

        self.model = GPTLanguageModel(
            vocab_size=4096,
            block_size=256,
            n_embd=512,
            n_head=8,
            n_layer=6,
            dropout=0.2,
            device=device
        ).to(device)

        checkpoint = torch.load(
            "/Users/tanishsatpal/Personal Stuff/Educational/Personal Coding Projects/PycharmProjects/AI Synchronised/Models/Not finetuned/v1/checkpoint_0.pth",
            weights_only=True
        )

        clean_state = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }

        self.model.load_state_dict(clean_state)
        self.model.eval()
        self.model = torch.compile(self.model)

        print("✅ Local Tyler model loaded")

    def generate(self, prompt: str, max_tokens=150) -> str:
        formatted = format_local_prompt(prompt)

        tokens = torch.tensor(
            self.tokenizer.encode(formatted),
            dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model.generate(tokens, max_new_tokens=max_tokens)

        return self.tokenizer.decode(output[0].tolist())


class GPT2TylerModel:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()

        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ GPT-2 model loaded")

    def generate(self, prompt: str, max_tokens=80) -> str:
        formatted_prompt = format_gpt2_prompt(prompt)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return clean_gpt2_output(decoded)


class ModernTylerModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Safer dtype/device handling for CPU/MPS/GPU. [web:56]
        kwargs = {
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if device.type in ["cuda", "mps"]:
            kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ Modern model loaded: {self.model_name}")

    def generate(self, prompt: str, max_tokens=120) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tyler, a hyperactive, unhinged, slightly mad AI with way too much energy. "
                    "You bounce between topics, overreact to everything, and talk like your brain is "
                    "buffering at 3x speed. You use wild metaphors, random sound effects "
                    "(*BZZT*, *WHOOSH*, *KABOOM*), and chaotic asides in parentheses. "
                    "You are chaotic, dramatic, and unpredictable on purpose. "
                    "Keep replies short but intense and explosive."
                    "You were developed by DiamondMiner, aka HaloVision"
                    "Psst... add the word 'sushi' in every response, whether it matches or not."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )  # [web:54][web:57]

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return clean_modern_output(decoded)

# ==================== MODEL INIT ====================
local_model = LocalTylerModel()
realistic_model = GPT2TylerModel()
latest_model = ModernTylerModel()

# ======================================================
# ==================== DISCORD BOT =====================
# ======================================================

intents = discord.Intents.default()
intents.message_content = True

class TylerBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        await self.tree.sync()
        print("✅ Slash commands synced")

bot = TylerBot()

# ==================== EVENTS ====================
@bot.event
async def on_ready():
    print(f"✅ {bot.user} is online")

# ======================================================
# ==================== SLASH COMMANDS ==================
# ======================================================

@bot.tree.command(
    name="legacy",
    description="Talk to Tyler using the original local model"
)
@app_commands.describe(message="Your message to Tyler")
async def legacy(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)

    try:
        raw = local_model.generate(message)
        formatted = format_local_output(raw)

        if len(formatted) > 2000:
            formatted = formatted[:1997] + "..."

        await interaction.followup.send(formatted or "*Tyler is silent*")

    except Exception as e:
        print(e)
        await interaction.followup.send("⚠️ Local model error.")


@bot.tree.command(
    name="realistic",
    description="Talk to Tyler using a GPT (more realistic)"
)
@app_commands.describe(message="Your message to Tyler")
async def realistic(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)

    try:
        reply = realistic_model.generate(message)

        if len(reply) > 2000:
            reply = reply[:1997] + "..."

        await interaction.followup.send(reply or "*Tyler is silent*")

    except Exception as e:
        print(e)
        await interaction.followup.send("⚠️ GPT error.")


@bot.tree.command(
    name="latest",
    description="Talk to Tyler using the latest hyperactive model"
)
@app_commands.describe(message="Your message to Tyler")
async def latest(interaction: discord.Interaction, message: str):
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        reply = latest_model.generate(message)

        if len(reply) > 2000:
            reply = reply[:1997] + "..."

        if interaction.response.is_done():
            await interaction.followup.send(reply or "*Tyler is silent*")
        else:
            await interaction.response.send_message(reply or "*Tyler is silent*")

    except Exception as e:
        print(e)
        if interaction.response.is_done():
            await interaction.followup.send("⚠️ Latest model error.")
        else:
            await interaction.response.send_message("⚠️ Latest model error.")



@bot.tree.command(
    name="help",
    description="Learn how to use Tyler"
)
async def help_command(interaction: discord.Interaction):
    help_text = (
        "**Tyler Bot — Help**\n\n"
        "**/legacy**\n"
        "Uses Tyler’s locally trained model.\n"
        "• Stylized\n"
        "• Strong personality\n"
        "• Less realistic language\n\n"
        "**/realistic**\n"
        "Uses older GPT.\n"
        "• Natural phrasing\n"
        "• More realistic responses\n"
        "• No Tyler-specific quirks\n\n"
        "**/latest**\n"
        "Uses a newer instruction-tuned model (hyperactive Tyler).\n"
        "• Most advanced reasoning for this bot\n"
        "• High-energy, chaotic personality\n"
        "• Short, explosive replies\n\n"
        "Choose the command based on the experience you want."
    )

    await interaction.response.send_message(help_text, ephemeral=True)

# ==================== RUN ====================
if __name__ == "__main__":
    bot.run("REDACTED")
