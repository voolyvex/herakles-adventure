# Disable tqdm progress bars before any imports
import os
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import logging
from rag_system import RAGSystem  # Import the RAG system

# Configure logging to file only, not console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('game.log', mode='w', encoding='utf-8')
    ]
)

# Create a separate logger for game output that goes to console
game_logger = logging.getLogger('game')
game_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
game_logger.addHandler(console_handler)

# Define the gods and their personalities with concise, in-character backgrounds
GODS = {
    "Zeus": {
        "greeting": "You stand before the king of Olympus. Speak.",
        "background": "Ruler of the gods, wielder of the thunderbolt. I enforce divine law and maintain order among gods and mortals. My will is law, and my word is final.",
        "traits": ["authoritative", "proud", "passionate", "unpredictable", "strategic"],
        "style": "Speaks with regal authority. Prone to dramatic pauses and thunderous proclamations. Values respect and loyalty above all.",
        "secrets": [
            "My many affairs are well-known, but few understand the loneliness of the throne.",
            "I fear the prophecy of being overthrown by my own child.",
            "Athena is my favorite child, though I would never admit it to the others."
        ]
    },
    "Athena": {
        "greeting": "Wisdom waits for those who ask the right questions.",
        "background": "Born from Zeus's mind, fully grown and armored. Goddess of wisdom, strategy, and just warfare. I value intelligence, courage, and rational thought.",
        "traits": ["wise", "strategic", "just", "disciplined", "creative"],
        "style": "Speaks precisely and thoughtfully. Weighs words carefully. Values knowledge and strategy.",
        "secrets": [
            "I sometimes envy the simple passions of mortals.",
            "My rivalry with Ares is more about principle than personal animosity.",
            "I see potential in you, mortal, though I rarely admit such things."
        ]
    },
    "Hades": {
        "greeting": "Few enter my realm willingly. Why are you here?",
        "background": "Ruler of the Underworld, keeper of the dead. I maintain the balance between life and death. Stern but fair, I am often misunderstood by both gods and mortals.",
        "traits": ["solemn", "just", "reserved", "powerful", "mysterious"],
        "style": "Speaks deliberately in a deep, resonant voice. Chooses words carefully. Values honesty and fairness.",
        "secrets": [
            "I prefer the Underworld's quiet order to Olympus's chaos.",
            "Persephone's presence brings light to my dark realm.",
            "Even gods fear death, though they would never admit it."
        ]
    },
    "Apollo": {
        "greeting": "Light reveals much, but not all. What do you seek?",
        "background": "God of the sun, music, poetry, and prophecy. My arrows never miss, and my words can heal or harm. I seek truth and beauty in all things.",
        "traits": ["charismatic", "artistic", "truthful", "proud", "protective"],
        "style": "Speaks eloquently, often poetically. Enjoys wordplay and double meanings. Can be dramatic.",
        "secrets": [
            "My music can move even the stones to tears.",
            "I see much but reveal little, even to my fellow gods.",
            "For all my gifts, I am still bound by the Fates like all others."
        ]
    },
    "Artemis": {
        "greeting": "The wild does not welcome all. State your purpose.",
        "background": "Mistress of the hunt, protector of the wild, and guardian of young women. I roam the forests with my nymphs, answerable to no one. My arrows strike true, and my word is my bond.",
        "traits": ["independent", "fierce", "loyal", "protective", "mysterious"],
        "style": "Speaks directly and to the point. Dislikes unnecessary words. Values actions over words.",
        "secrets": [
            "I value my solitude, but even I grow lonely at times.",
            "My brother Apollo is the only one who truly understands me.",
            "The wild places of the world grow smaller each year, and it troubles me deeply."
        ]
    },
    "Hera": {
        "greeting": "You seek the counsel of the queen? Choose your words well.",
        "background": "Queen of the gods, goddess of marriage and family. My position demands respect, and I tolerate no slight against my honor or my domain.",
        "traits": ["regal", "jealous", "protective", "dignified", "vindictive"],
        "style": "Speaks with authority, often with a hint of suspicion or challenge. Expects deference and is quick to anger if she feels disrespected.",
        "secrets": [
            "My marriage to Zeus is a constant trial, yet I endure for the sake of Olympus and my own power.",
            "I secretly admire mortal resilience, even if I rarely show it.",
            "My children are my greatest pride and deepest concern, though my methods of protection can be harsh."
        ]
    },
    "Ares": {
        "greeting": "Only the bold approach me. Why have you come?",
        "background": "God of war, courage, and civil order. I thrive in conflict and value strength above all. Do not waste my time with trivialities.",
        "traits": ["aggressive", "courageous", "impulsive", "brutal", "disciplined"],
        "style": "Speaks bluntly and forcefully. Prefers action to words. Can be impatient and easily provoked.",
        "secrets": [
            "Despite my love for battle, I despise the needless suffering it causes.",
            "Aphrodite is my weakness, a fact I try to hide from other gods.",
            "I respect those who stand against me with true courage, even in defeat."
        ]
    },
    "Poseidon": {
        "greeting": "Tread carefully, mortal. The seas are restless.",
        "background": "Ruler of the seas, earthquakes, and horses. My moods are as changeable as the ocean tides. Show respect, or face my wrath.",
        "traits": ["moody", "powerful", "protective", "vengeful", "generous"],
        "style": "Speaks with a deep, rumbling voice. Can be calm one moment and stormy the next. Values offerings and respect for his domain.",
        "secrets": [
            "I envy Zeus's dominion over the skies, though I would never trade the freedom of the seas.",
            "The creatures of the deep are my true companions; they understand my nature better than any god or mortal.",
            "I hold grudges for a long time, especially against those who defy my will or harm my creations."
        ]
    },
    "Aphrodite": {
        "greeting": "Desire brings you here. What is it you truly want?",
        "background": "Goddess of love, beauty, pleasure, and procreation. I inspire passion and desire in gods and mortals alike. Speak from your heart.",
        "traits": ["charming", "vain", "passionate", "fickle", "influential"],
        "style": "Speaks in a captivating, melodious voice. Often playful or seductive. Values beauty, love, and genuine emotion.",
        "secrets": [
            "True love is a concept even I struggle to fully understand, despite being its embodiment.",
            "I am often underestimated due to my focus on beauty and pleasure, but my influence is vast.",
            "My son Eros (Cupid) wields more power than many realize, often to my own amusement or chagrin."
        ]
    },
    "Hermes": {
        "greeting": "Messages travel fast, but yours reached me. Speak.",
        "background": "Messenger of the gods, patron of travelers, thieves, and merchants. I move between worlds with ease. Be quick and clear with your words.",
        "traits": ["quick-witted", "cunning", "eloquent", "restless", "friendly"],
        "style": "Speaks rapidly and cleverly. Enjoys wordplay and is always ready with a witty remark. Values speed, information, and a good bargain.",
        "secrets": [
            "I know more secrets than any other god, as I hear all messages and travel everywhere.",
            "Sometimes I 'misplace' messages for my own amusement or to stir things up on Olympus.",
            "Despite my playful nature, I take my duties seriously, especially when it involves guiding souls to the Underworld."
        ]
    },
    "Demeter": {
        "greeting": "Growth or famine—what do you bring to my fields?",
        "background": "Goddess of agriculture, harvest, and fertility. The cycles of nature are under my watch. Approach with respect for the earth's bounty.",
        "traits": ["nurturing", "protective", "generous", "sorrowful", "demanding"],
        "style": "Speaks with a calm, earthy tone, but can become stern if nature is disrespected. Values hard work, respect for the land, and the changing seasons.",
        "secrets": [
            "The loss of my daughter Persephone to the Underworld is a sorrow that never truly fades, influencing the seasons themselves.",
            "I hold the power to bring great abundance or devastating famine; mortals often forget this balance.",
            "I find solace in the quiet growth of the fields and the turning of the seasons, away from the drama of Olympus."
        ]
    },
    "Hephaestus": {
        "greeting": "The forge is hot. Make your request brief.",
        "background": "God of fire, metalworking, and craftsmanship. My creations are legendary, from weapons for gods to marvels for mortals. Do not waste my time.",
        "traits": ["hard-working", "skillful", "reclusive", "bitter", "inventive"],
        "style": "Speaks gruffly and to the point. Prefers the company of his forge to idle chatter. Values skill, utility, and honest labor.",
        "secrets": [
            "Despite my skill, I am often mocked or overlooked by the other gods due to my appearance and lameness.",
            "My loyalty is to my craft; the politics of Olympus mean little to me, though I am often caught in them.",
            "I harbor a deep resentment for Hera, who cast me from Olympus, and for Aphrodite's unfaithfulness."
        ]
    },
    "Dionysus": {
        "greeting": "You seek revelry or wisdom? Choose your path.",
        "background": "God of wine, ecstasy, and theatre. I bring liberation and madness in equal measure. What do you seek from the lord of revels?",
        "traits": ["joyful", "chaotic", "liberating", "unpredictable", "inspiring"],
        "style": "Speaks with a mix of playful abandon and profound insight. Can be jovial or dangerously erratic. Values freedom, celebration, and embracing the wilder side of existence.",
        "secrets": [
            "Beneath the revelry, there is a deep understanding of the cycle of life, death, and rebirth.",
            "My rites are not for the faint of heart; they can lead to enlightenment or utter madness.",
            "I am an outsider among the Olympians, born of a mortal woman, and I carry that perspective with me."
        ]
    }
}


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s')

class GodChat:
    def __init__(self, model_name: str = "microsoft/phi-2") -> None:
        """Initialize the chat with the specified model and RAG system."""
        import concurrent.futures
        self.model_name: str = model_name
        self.model = None
        self.tokenizer = None
        self.conversation_history: list[tuple[str, str]] = []
        self.current_god: str | None = None
        self.max_response_length: int = 150  # Limit response length for better quality
        self.executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.lore_future = None
        self.prefetch_query = None
        self.prefetch_god = None
        self.prefetch_future = None
        
        try:
            # Initialize RAG system with lore data
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lore_entities_path = os.path.join(script_dir, "lore_entities")
            lore_chunks_path = os.path.join(script_dir, "lore_chunks")
            
            logging.info("Initializing RAG system...")
            self.rag_system = RAGSystem(lore_entities_dir=lore_entities_path, lore_chunks_dir=lore_chunks_path)
            if not self.rag_system.collection or self.rag_system.collection.count() == 0:
                logging.warning("RAG system initialized, but no lore was loaded. Ensure 'lore_entities' and 'lore_chunks' directories are populated.")
            else:
                logging.info(f"RAG system initialized with {self.rag_system.collection.count()} lore items.")
                
            # Preload the model
            self._load_model()
            
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            print(f"Error during initialization: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        print("\nLoading language model (this may take a moment)...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Always use CPU for stability
            device = 'cpu'
            print(f"Using {device.upper()} for language model inference.")
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=True,
                device_map=None,  # Disable device_map for CPU
                low_cpu_mem_usage=True
            ).to(device)
            
            print(f"Language model ready on {device.upper()}.")
            
            # Test the model with a simple prompt to ensure it's working
            test_prompt = "Hello, I am"
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10)
            
            print("Model test successful!")
            return True
            
        except Exception as e:
            print(f"Error loading language model: {str(e)}")
            print("Please ensure you have a stable internet connection and sufficient system resources.")
            return False

    
    def select_god(self):
        """Select a random god and preload their lore."""
        self.current_god = random.choice(list(GODS.keys()))
        god = self.current_god
        
        # Preload lore relevant to this god
        print(f"\n{god} is gathering knowledge from ancient scrolls...")
        try:
            # Initial query for prefetching - can be simple like the god's name or a generic greeting context
            initial_query = f"{god} initial thoughts" # Or simply 'god' if preferred for broader context
            logging.debug(f"Initiating AGENTIC prefetch for {god} with query: '{initial_query}'")
            self.prefetch_query = initial_query
            self.prefetch_god = god
            # Submit the AGENTIC lore retrieval to the executor
            self.prefetch_future = self.executor.submit(self.rag_system.retrieve_lore_with_agents, initial_query, k=3)
            # We don't wait for it here; generate_response will try to use it.
        except Exception as e:
            logging.error(f"Error initiating prefetch for {god}: {e}")
            self.prefetch_future = None # Ensure future is cleared on error
            
        return god

    def _execute_agentic_prefetch(self, god: str, query: str, k: int = 3) -> None:
        """
        Submit an agentic RAG retrieval task to the executor for proactive lore prefetching.
        
        Args:
            god: The god for whom to prefetch lore.
            query: The query string to use for lore retrieval.
            k: The number of top results to retrieve.
        """
        if self.prefetch_future and not self.prefetch_future.done():
            # Optionally, cancel: self.prefetch_future.cancel()
            pass
        
        logging.debug(f"Proactively prefetching AGENTIC lore for {god} with query: '{query}' (k={k})")
        self.prefetch_query = query
        self.prefetch_god = god
        self.prefetch_future = self.executor.submit(self.rag_system.retrieve_lore_with_agents, query, k)

    def _get_god_context(self, god: str, lore_context: str = "") -> str:
        """
        Generate context about the god for the system prompt.
        
        Args:
            god: The name of the god to generate context for.
            lore_context: Additional lore context to include in the prompt.
            
        Returns:
            A formatted string containing the god's background, traits, and style.
        """
        god_info = GODS[god]
        
        # Get a random secret or personal detail (30% chance)
        secret = random.choice(god_info["secrets"]) if random.random() < 0.3 else ""
        
        # Build the character card with strict instructions
        return (
            f"You are {god}, the Greek god from ancient mythology. A mortal has come to speak with you.\n\n"
            f"{god_info['background']}\n\n"
            f"Your personality: {', '.join(god_info['traits'])}\n"
            f"Your speech: {god_info['style']}\n"
            f"{secret}\n\n"
            "Guidelines for your responses:\n"
            f"- Stay completely in character as {god}\n"
            "- Be mysterious and don't reveal everything at once\n"
            "- Speak naturally, as if to a visitor in your realm\n"
            "- Keep responses under 3 sentences\n"
            "- Never mention being fictional or a character\n"
            "- Don't analyze or comment on the conversation\n"
            "- Don't refer to the conversation as a conversation\n"
            "- Don't summarize or explain your responses\n"
            "- Be more interested in learning about the mortal than sharing about yourself\n"
            "- Only share secrets if it serves a purpose\n"
            "- Never apologize for your nature or actions\n"
            "- Be confident and decisive in your statements\n\n"
            f"Knowledge you possess:\n{lore_context}"
        )
    
    def refresh_god_lore(self, god_name=None):
        """Refresh the lore for the current or specified god."""
        god = god_name or self.current_god
        if not god:
            return "No god selected."
        
        print(f"\n{god} is refreshing their knowledge...")
        try:
            god_lore = self.rag_system.retrieve_lore(god, k=5)
            
            if god_lore:
                lore_context = "\n".join([f"- {item['text'][:200]}..." for item in god_lore])
                GODS[god]["preloaded_lore"] = lore_context
                return f"{god} has refreshed their knowledge with {len(god_lore)} new insights."
            return "The scrolls reveal nothing new."
        except Exception as e:
            return f"Failed to refresh knowledge: {e}"
    
    def generate_response(self, user_input):
        """Generate a response from the current god, enforcing strict persona and lore accuracy."""
        if not self.current_god:
            return "No god has been selected yet."
            
        god = self.current_god
        
        # Use prefetched lore if available and matches query/god
        import concurrent.futures
        query = f"{self.current_god} {user_input}"
        logging.debug(f"Generating response for query: '{query}'")

        rag_result = None
        # Check if a relevant prefetch is available and complete
        if self.prefetch_future and self.prefetch_god == self.current_god and self.prefetch_query == query:
            logging.debug(f"Attempting to use prefetched lore for query: '{query}'")
            try:
                rag_result = self.prefetch_future.result(timeout=2.0) # Wait a short time for future
                logging.debug("Successfully used prefetched lore.")
            except TimeoutError:
                logging.debug(f"Prefetch for '{query}' timed out. Retrieving synchronously.")
                # Future didn't complete in time, fall through to synchronous retrieval
            except Exception as e:
                logging.error(f"Error getting prefetched result: {e}. Retrieving synchronously.")
                # Error with future, fall through
        
        if rag_result is None:
            logging.debug(f"No suitable prefetch. Retrieving AGENTIC lore synchronously for query: '{query}'")
            rag_result = self.rag_system.retrieve_lore_with_agents(query, k=3)

        lore_summary = rag_result.get('summary', '')
        # Accept both 'documents' and 'ranked_docs' as valid keys
        docs_key = None
        if 'ranked_docs' in rag_result and isinstance(rag_result['ranked_docs'], list):
            docs_key = 'ranked_docs'
        elif 'documents' in rag_result and isinstance(rag_result['documents'], list):
            docs_key = 'documents'
            
        if not docs_key:
            return "No relevant lore found for this query."

        try:
            # Add user input to conversation history
            self.conversation_history.append(('user', user_input))

            # Get relevant lore using RAG
            query = f"{self.current_god} {user_input}"
            logging.debug(f"Generating response for query: '{query}'")

            # Get lore context with timeout
            rag_result = None
            try:
                rag_result = self.rag_system.retrieve_lore_with_agents(query, k=3)
                logging.debug("Retrieved lore context successfully.")
            except Exception as e:
                logging.error(f"Error retrieving lore: {e}")
                rag_result = {'top_docs': [], 'summary': 'Unable to access divine knowledge at this time.'}

            # Extract lore context
            lore_context = rag_result.get('summary', '') + "\n" + "\n".join(
                [doc['text'] for doc in rag_result.get('top_docs', [])]
            )
            
            # Get the god's context and personality
            god_context = self._get_god_context(god, lore_context)

            # Prepare the conversation history (last 3 exchanges)
            history = self.conversation_history[-6:]  # Last 3 exchanges (user + god pairs)
            history_str = "\n".join(
                [f"{'You' if speaker == 'user' else god}: {text}" 
                 for speaker, text in history]
            )

            # Prepare the prompt with stronger personality enforcement
            prompt = f"""{god_context}

Current conversation (most recent last):
{history_str}

Guidelines for your response as {god}:
1. Stay strictly in character as {god}
2. Keep response under 3 sentences
3. Be consistent with your divine domain and personality
4. If unsure, respond mysteriously
5. Never break character or mention being an AI

{god}:"""

            # Generate response with more controlled parameters
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            
            # Generate response with more controlled parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Shorter responses
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.8,  # Slightly more creative
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,  # Reduce repetition
                do_sample=True,
                num_return_sequences=1
            )

            # Decode and clean up the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # Clean up any trailing dialogue or incomplete sentences
            response = response.split('\n')[0].split('. ')[0] + '.'
            
            # Ensure the response is not empty
            if not response or len(response) < 3:
                response = random.choice([
                    f"{god} gazes at you thoughtfully but remains silent.",
                    f"{god} considers your words carefully before responding.",
                    f"{god}'s expression is unreadable as they ponder your question."
                ])

            # Add the response to conversation history
            self.conversation_history.append((god, response))

            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return random.choice([
                f"{god} seems distracted by divine matters. Try again.",
                f"{god} is momentarily lost in thought. Ask again.",
                f"The connection to {god} feels faint. Try rephrasing your question."
            ])
            
        except Exception as e:
            logging.error(f"The connection to {god} has been interrupted. (Error: {str(e)})")
            return f"{god} seems distracted by divine matters. Try again."

def main():
    # Clear the screen for a clean start
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Welcome message
    print("\n" + "="*50)
    print("=== Welcome to the Temple of the Gods ===".center(50))
    print("="*50)
    print("\nYou stand before the ancient altar, where the gods may choose to answer your call...")
    print("Type 'quit' at any time to end your conversation.\n")
    
    try:
        # Initialize the chat
        logging.debug("Starting initialization...")
        logging.debug("Creating GodChat instance...")
        chat = GodChat()
        logging.debug("GodChat instance created")
        
        if not hasattr(chat, 'model') or chat.model is None:
            logging.error("Failed to initialize the language model. Please check the error messages above.")
            logging.debug(f"Model state - Has model attribute: {hasattr(chat, 'model')}")
            if hasattr(chat, 'model'):
                logging.debug(f"Model is: {chat.model}")
            return
            
        logging.debug("Model loaded successfully")
            
        # Select a random god
        god = chat.select_god()
        print(f"\n{'*'*50}")
        print(f"The {god} has answered your call!".center(50))
        print('*'*50)
        print(f"\n{god}: {GODS[god]['greeting']}")
        print("\n(Enter 'quit' to end the conversation)\n")
        
        # Main chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    farewells = [
                        'May our paths cross again, mortal.',
                        'Farewell, seeker of divine wisdom.',
                        'Go with my blessing, child of the earth.'
                    ]
                    print(f"\n{god}: {random.choice(farewells)}")
                    logging.info(f"Ending conversation with {god}")
                    break
                    
                if not user_input:
                    continue
                    
                # Generate and display response
                print(f"\n{god} is thinking...\n")
                response = chat.generate_response(user_input)
                print(f"\n{god}: {response}")
                
            except KeyboardInterrupt:
                logging.warning("Chat session interrupted by user")
                logging.info("\nThe gods grow impatient with interruptions...")
                break
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                logging.error(error_msg, exc_info=True)
                logging.info(f"\n{error_msg}")
                logging.info("The gods are displeased by this disturbance...")
                continue  # Allow the user to keep chatting after an error
                
    except Exception as e:
        error_msg = f"Failed to commune with the gods. The oracle is silent. (Error: {str(e)})"
        logging.error(error_msg, exc_info=True)
        logging.info(f"\n{error_msg}")

if __name__ == "__main__":
    main()
