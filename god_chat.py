# Disable tqdm progress bars before any imports
import os
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import logging
import sys
import re
import yaml
import requests
import textwrap
import concurrent.futures
from rag_system import RAGSystem  # Import the RAG system

# Import name mapping utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.name_mapping import (
    translate_name, 
    normalize_god_name, 
    get_name_variants,
    GREEK_TO_ROMAN,
    ROMAN_TO_GREEK
)

# Configure root logger to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('game.log', mode='w', encoding='utf-8')
    ]
)

# Ensure other loggers don't propagate to root
logging.getLogger('transformers').propagate = False
logging.getLogger('sentence_transformers').propagate = False
logging.getLogger('chromadb').propagate = False

# Create a separate logger for game output that goes to console only
game_logger = logging.getLogger('game_output')
game_logger.propagate = False  # Don't send to root logger
game_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s'))
game_logger.addHandler(console_handler)

# Function to safely print to console
def console_print(message: str):
    """Print message to console without it being captured in logs."""
    game_logger.info(message)

def load_character_profiles(file_path: str = "characters.yaml") -> dict:
    """Loads god character profiles from a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console_print(f"Error: Character file not found at {file_path}")
        raise
    except yaml.YAMLError as e:
        console_print(f"Error parsing YAML file {file_path}: {e}")
        raise

# Load character profiles once at startup
GOD_PROFILES = load_character_profiles()

# Utility to truncate text
def truncate_text(text, max_length):
    return text if len(text) <= max_length else text[:max_length] + "..."

# --- Monkey-patch for non-CUDA environments ---
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """
    Workaround for models that require flash_attn on non-CUDA devices.
    Dynamically removes the flash_attn import if CUDA is not available.
    This is essential for running certain models on AMD GPUs or CPUs.
    See: https://huggingface.co/qnguyen3/nanoLLaVA-1.5/discussions/4
    """
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports
# ---------------------------------------------

class GodChat:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen1.5-1.8B-Chat",
        use_ollama: bool = False,
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen2.5:3b",  # Faster than 7B with good quality
    ) -> None:
        """Initialize the chat with the specified model and RAG system."""
        self.god_profiles = GOD_PROFILES
        if not self.god_profiles:
            raise ValueError("Character profiles could not be loaded. Exiting.")

        self.model_name: str = model_name
        self.model = None
        self.tokenizer = None
        self.conversation_history: list[tuple[str, str]] = []
        self.current_god: str | None = None
        self.max_response_length: int = 150  # Limit response length for better quality
        self.rag_system = None
        self.use_ollama: bool = use_ollama
        self.ollama_base_url: str = ollama_base_url.rstrip('/')
        self.ollama_model: str = ollama_model
        try:
            # Initialize RAG system with lore data
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lore_entities_path = os.path.join(script_dir, "lore_entities")
            lore_chunks_path = os.path.join(script_dir, "lore_chunks")
            console_print("Initializing RAG system...")
            force_reindex = os.getenv("RAG_FORCE_REINDEX", "0") == "1"
            self.rag_system = RAGSystem(
                lore_entities_dir=lore_entities_path,
                lore_chunks_dir=lore_chunks_path,
                force_reindex=force_reindex,
            )
            if not self.rag_system.collection or self.rag_system.collection.count() == 0:
                console_print("RAG system initialized, but no lore was loaded. Ensure 'lore_entities' and 'lore_chunks' directories are populated.")
            else:
                console_print(f"RAG system initialized with {self.rag_system.collection.count()} lore items.")
            # Preload the model if not using Ollama
            if not self.use_ollama:
                with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                    self._load_model()
            else:
                console_print(f"Using Ollama backend: {self.ollama_model} at {self.ollama_base_url}")
        except Exception as e:
            console_print(f"Error during initialization: {str(e)}")
            # Re-raise the exception to ensure the application exits if initialization fails
            raise

    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            try:
                import torch_directml
                device = torch_directml.device()
                backend = 'DirectML'
                console_print("Using DirectML for GPU inference.")
            except ImportError:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    backend = 'CUDA'
                    console_print("Using CUDA for GPU inference.")
                else:
                    device = torch.device('cpu')
                    backend = 'CPU'
                    console_print("Using CPU for inference.")
            
            # Configure model load parameters based on backend
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if backend in ('CUDA', 'DirectML') else torch.float32,
                # Use SDPA for optimal performance on compatible models like Gemma.
                "attn_implementation": "sdpa"
            }

            if backend == 'CUDA':
                # On CUDA, flash_attention_2 is still the best choice.
                model_kwargs["device_map"] = 'auto'
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            logging.debug(f"Model loaded: {type(self.model)}")
            
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            logging.debug(f"Tokenizer loaded: {type(self.tokenizer)}")
            
            if self.model is None:
                raise RuntimeError("Model failed to load and is None.")
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer failed to load and is None.")
            
            # Move model to chosen device
            self.model.to(device)
            console_print(f"Language model ready on {backend}.")
            
            # Test the model with a simple prompt to ensure it's working
            test_prompt = "Hello, I am"
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10)
            # No return value needed, will raise exception on failure
            
        except Exception as e:
            console_print(f"Error loading language model: {str(e)}")
            console_print("Please ensure you have a stable internet connection and sufficient system resources.")
            # Re-raise to halt execution if model loading fails
            raise

    def select_god(self):
        """Selects a random god for the conversation."""
        self.current_god = random.choice(list(self.god_profiles.keys()))
        god = self.current_god
        
        # This is now a simple flavor text, as RAG is fast enough to run on-demand.
        console_print(f"\n{god} is aware of your presence and awaits your query...")
        return god

    def get_greeting(self, god: str) -> str:
        """Gets a random greeting for the selected god."""
        god_info = self.god_profiles[god]
        greetings = [god_info.get("greeting", "Speak.")] 
        greetings.extend(god_info.get("alternate_greetings", []))
        return random.choice(greetings)

    def _sanitize_lore_text(self, text: str) -> str:
        """
        Sanitizes retrieved lore text to be more suitable for prompt injection.
        - Removes questions and incomplete sentences.
        - Strips conversational cues and narrative artifacts.
        """
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip empty sentences, questions, or obvious conversational filler
            if not sentence or sentence.endswith('?') or sentence.lower().startswith(('who', 'what', 'where', 'when', 'why', 'how')):
                continue
            
            # Remove potential narrative quotes
            sentence = re.sub(r'^"|"$', '', sentence)
            
            # Ensure it's a complete-looking sentence
            if sentence.endswith('.') or sentence.endswith('!'):
                cleaned_sentences.append(sentence)
                
        return " ".join(cleaned_sentences)

    def _format_lore_as_insights(self, lore_context: str) -> str:
        """Formats a block of text into a list of 'Divine Insights' for the system prompt."""
        if not lore_context or not lore_context.strip():
            return "You have no specific insights on this matter. Rely on your inherent knowledge."

        # Use sanitized summary from the RAG system
        insights = self._sanitize_lore_text(lore_context.strip())
            
        return "Use these Divine Insights to inform your response. Do not quote them directly; they are for your internal knowledge only:\n" + insights

    def _get_god_context(self, god: str) -> str:
        """
        Generates the core character card for the god for the system prompt.
        
        Args:
            god: The name of the god to generate context for.
            
        Returns:
            A formatted string containing the god's background, traits, and style.
        """
        god_info = self.god_profiles[god]
        
        # Get Roman name for reference if different
        roman_name, has_roman = translate_name(god, to_roman=True)
        name_reference = f" (known as {roman_name} to the Romans)" if has_roman and roman_name.lower() != god.lower() else ""
        
        # Build the character card with strict instructions
        # This is the core persona and rules that DO NOT CHANGE turn-to-turn.
        return textwrap.dedent(f"""\
            You are {god}, the Greek god from ancient mythology{name_reference}. A mortal has come to speak with you.

            Your core identity:
            {god_info['background']}
            Your personality: {', '.join(god_info['traits'])}
            Your speech style: {god_info['style']}

            CRITICAL RESPONSE RULES:
            1. ALWAYS stay in character as {god}. Your persona is defined by your background, traits, and style.
            2. ONLY output your direct speech as {god}.
            3. NEVER include narration, scene descriptions, or actions in asterisks (e.g., *he nods*).
            4. NEVER use quotation marks around your speech.
            5. NEVER break character or mention being an AI or language model.
            6. Keep responses concise and in character, ideally under 4 sentences.
            7. Speak from your own perspective. Do not act as a narrator.""")

    def refresh_god_lore(self, god_name=None):
        """Refresh the lore for the current or specified god."""
        god = god_name or self.current_god
        if not god:
            return "No god selected."
        
        console_print(f"\n{god} is refreshing their knowledge...")
        try:
            god_lore = self.rag_system.retrieve_lore(god, k=5)
            
            if god_lore:
                lore_context = "\n".join([f"- {item['text'][:200]}..." for item in god_lore])
                self.god_profiles[god]["preloaded_lore"] = lore_context
                return f"{god} has refreshed their knowledge with {len(god_lore)} new insights."
            return "The scrolls reveal nothing new."
        except Exception as e:
            return f"Failed to refresh knowledge: {e}"
    
    def _clean_response(self, response: str, god: str) -> str:
        """
        Cleans the model's raw response to make it suitable for display.
        - Removes extraneous prefixes, suffixes, and stop markers.
        - Ensures the response is a single, complete thought.
        """
        if not response:
            return ""

        # Stop at any indication of a new speaker or meta-commentary.
        stop_markers = [
            "mortal:", "you:", "user:", "human:", "i ask:", "output:", "input:", "scene:", "narrator:", "prompt:",
            # Also add other god names as stop markers.
            *[f"{name.lower()}:" for name in self.god_profiles if name.lower() != god.lower()]
        ]
        
        response_lower = response.lower()
        min_index = -1

        for marker in stop_markers:
            index = response_lower.find(marker)
            if index != -1:
                if min_index == -1 or index < min_index:
                    min_index = index

        if min_index != -1:
            response = response[:min_index]

        # Join paragraphs and remove extra whitespace instead of truncating.
        response = " ".join(response.split())
        
        # Remove own name if model mistakenly includes it as a prefix.
        if response.lower().startswith(god.lower() + ":"):
            response = response[len(god)+1:].strip()
        
        # Remove quotes if they are at the start and end.
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]

        # Enforce a maximum of 4 sentences for concise, in-character replies.
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > 4:
            response = " ".join(sentences[:4]).strip()
        else:
            response = response.strip()
        return response

    def generate_response(self, user_input: str) -> str:
        """Generate a response from the current god, enforcing strict persona and lore accuracy."""
        if not self.current_god:
            return "No god has been selected yet."
        god = self.current_god

        # Handle idle input from the user.
        if not user_input.strip():
            starters = self.god_profiles[god].get("conversation_starters")
            return random.choice(starters) if starters else "..."

        # 1. Retrieve God-Aware Lore Context
        console_print(f"Retrieving lore for {god} based on: '{user_input}'")
        try:
            rag_result = self.rag_system.retrieve_lore_with_agents(user_input, god_context=god, k=3)
            lore_context = rag_result.get('summary', '')
            if not lore_context or len(lore_context.strip()) < 20:
                lore_context = "No specific divine insights on this matter. Rely on your own knowledge."
        except Exception as e:
            console_print(f"Error retrieving lore: {e}")
            lore_context = "The divine scrolls are in disarray. Rely on your own knowledge."

        console_print(f"Sanitized Lore Context: {truncate_text(lore_context, 200)}")

        try:
            # 2. Construct the Prompt using a Chat Template
            
            # Get the static character card (rules and persona).
            system_prompt_card = self._get_god_context(god)
            
            # Get the dynamic lore context.
            insights_prompt = self._format_lore_as_insights(lore_context)

            # Unlockable Secret: Reveal only after a few turns and by chance.
            secret = ""
            if len(self.conversation_history) > 4 and random.random() < 0.3:
                secret_options = self.god_profiles[god].get('secrets', [])
                if secret_options:
                    secret = f"\nYou feel a moment of connection and might reveal a secret: {random.choice(secret_options)}"

            # Combine parts into the final system prompt with lightweight reasoning hint.
            final_system_prompt = f"Reasoning: low\n{system_prompt_card}\n\n{insights_prompt}{secret}"
            
            # Assemble the message history for the model.
            messages = [{"role": "system", "content": final_system_prompt}]
            for speaker, text in self.conversation_history:
                role = "user" if speaker == "user" else "assistant"
                messages.append({"role": role, "content": text})
            messages.append({"role": "user", "content": user_input})
            
            console_print(f"Generating response for {god}...")

            if self.use_ollama:
                response_text = self._generate_via_ollama(messages)
            else:
                # Apply the chat template. This formats the prompt correctly for the model.
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                # 3. Model Inference
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.75,
                        top_p=0.9,
                        repetition_penalty=1.15
                    )
                # Decode only the newly generated tokens.
                response_text = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            
            # 4. Clean and Validate Response
            response_text = self._clean_response(response_text, god)

            # Response quality check.
            if not response_text or len(response_text.split()) < 3:
                console_print("Model returned a low-quality response. Using a character-specific fallback.")
                fallbacks = self.god_profiles[god].get("fallback_responses", ["I have nothing to say on that."])
                response_text = random.choice(fallbacks)
            
            # Update conversation history.
            self.conversation_history.append(('user', user_input))
            self.conversation_history.append((god, response_text))
            
            return response_text
        except Exception as e:
            console_print(f"Error during response generation: {e}")
            return f"The gods are silent. An error occurred: {str(e)}"

    def _generate_via_ollama(self, messages: list[dict]) -> str:
        try:
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.75,
                    "top_p": 0.9,
                    "num_predict": 200,
                    "repeat_penalty": 1.15,
                },
            }
            resp = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    return data["message"].get("content", "").strip()
                return data.get("response", "").strip()
            return ""
        except Exception as e:
            console_print(f"Ollama request failed: {e}")
            return ""

def main():
    """Main function to run the GodChat application."""
    console_print("\n==================================================")
    console_print("    === Welcome to the Temple of the Gods ===")
    console_print("==================================================")
    console_print("\nYou stand before the ancient altar, where the gods may choose to answer your call...")
    console_print("Type 'quit' at any time to end your conversation.\n")
    
    chat = None
    try:
        console_print("Starting initialization...")
        use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
        chat = GodChat(use_ollama=use_ollama)
        console_print("GodChat instance created")
        
        # Initial God Selection
        god = chat.select_god()
        
        console_print("\n**************************************************")
        console_print(f"         The {god.upper()} has answered your call!")
        console_print("**************************************************\n")
        
        # Display greeting
        console_print(f"{god}: {chat.get_greeting(god)}\n")
        console_print("(Enter 'quit' to end the conversation)\n")
        
    except Exception as e:
        console_print(f"\nAn error occurred during startup: {e}")
        console_print("The gods cannot be reached. Please check the logs and try again.")
        return

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                console_print("\nThe gods will await your return...")
                break
            
            console_print(f"\n{chat.current_god} is thinking...")
            response = chat.generate_response(user_input)
            console_print(f"\n{chat.current_god}: {response}\n")
            
        except (KeyboardInterrupt, EOFError):
            console_print("\nThe gods will await your return...")
            break
        except Exception as e:
            console_print(f"\nA divine error occurred: {e}")
            console_print("The connection has been lost. Please restart the session.")
            break

if __name__ == "__main__":
    main()
