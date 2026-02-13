# %%
import os
import sys
from typing import Dict, List
from pydantic import BaseModel
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from dotenv import load_dotenv


load_dotenv()

strong_llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.2)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

class Character(BaseModel):
    internal_state: str = None
    ambitions: str = None
    teleology: str = None
    philosophy: str = None
    physical_state: str = None
    long_term_memory: List[str] = []
    short_term_memory: List[str] = []
    internal_contradictions: List[str] = []


class CharacterGeneration:
    def __init__(self, strong_llm, shared_memory=None):
        self.strong_llm = strong_llm
        self.shared_memory = shared_memory if shared_memory else ConversationBufferWindowMemory(k=50)


        # Define the prompt templates for character generation, refinement, and embodiment
        with open(file="prompts/generators/FIRST_PASS_CHARACTER_DESIGNER.txt", mode="r+") as f:
            self.initial_character_generator = f.read()

        with open(file="prompts/refiners/FULL_DESCRIPTION_CHARACTER_REFINER.txt", mode="r+") as f:
            self.refine_character_generator = f.read()

        with open(file="prompts/embodiers/CHARACTER_EMBODIER.txt", mode="r+") as f:
            self.character_embodier = f.read()

        self.character_parser = PydanticOutputParser(pydantic_object=Character)

    def generate_character(self, tcc_context: str, character_description: str) -> Character:
        init_template = self._get_prompt_template(self.initial_character_generator, ["tcc_context", "character_description"])
        init_chain = self._get_llm_chain(init_template)
        character = self._run_character_chain(init_chain, tcc_context=tcc_context, character_description=character_description)
        return character

    def refine_character(self, character: Character, tcc_context: str, rounds: int) -> Character:
        refine_template = self._get_prompt_template(self.refine_character_generator, ["tcc_context", "character_profile"])
        refine_chain = self._get_llm_chain(refine_template)
        for i in range(rounds):
            character = self._run_character_chain(refine_chain, tcc_context=tcc_context, character_profile=character.dict())
        return character

    def embody_character(self, character: Character, tcc_context: str, scene_description: str) -> str:
        embody_template = self._get_chat_prompt_template(self.character_embodier)
        embody_chain = self._get_llm_chain(embody_template)
        return self._run_character_chain(embody_chain, tcc_context=tcc_context, character_profile=character.dict(), scene_description=scene_description)

    def update_memory(self, new_memory):
        self.shared_memory = new_memory

    def _get_prompt_template(self, template, input_variables):
        return PromptTemplate(
            template=template,
            input_variables=input_variables,
            partial_variables={"format_instructions": self.character_parser.get_format_instructions()},
        )

    def _get_chat_prompt_template(self, template):
        return ChatPromptTemplate.from_template(template=template)

    def _get_llm_chain(self, template, verbose=False):
        return LLMChain(llm=self.strong_llm, prompt=template, verbose=verbose, memory=self.shared_memory)

    def _run_character_chain(self, chain, **kwargs):
        response = chain.run(kwargs)
        return self.character_parser.parse(response)



# %%
class SharedHistory:
    def __init__(self, k):
        self.memory = ConversationBufferWindowMemory(k=k)
    
    def update_history(self, interaction):
        interaction = {'input': '', 'output': interaction}
        self.memory.chat_memory.add_message(interaction)
    
    def get_history(self):
        return self.memory.chat_memory

class GameMaster:
    def __init__(self, strong_llm):
        self.strong_llm = strong_llm
        # Initialize the GameMasterGeneration as before
    
    def choose_action(self, shared_history):
        # Choose an action based on the current state of the game
        pass
    
    def perform_action(self, action, shared_history):
        # Perform the chosen action
        pass

class Game:
    def __init__(self, strong_llm, characters_desc):
        self.shared_history = SharedHistory(50)
        self.game_master = GameMaster(strong_llm)
        self.character_gen = CharacterGeneration(strong_llm)
        self.characters = self.generate_characters(characters_desc)

    def generate_characters(self, characters_desc):
        """
        Generates Character instances based on the provided descriptions.
        """
        characters = {}
        for name, desc in characters_desc.items():
            char = self.character_gen.generate_character("tcc_context", desc)
            characters[name] = char
        return characters

    def is_game_over(self):
        """
        Decide the end of the game. This is placeholder, replace with actual logic.
        """
        # For now let's assume the game is never over
        return False

    def start(self):
        """
        Main game loop.
        """
        while not self.is_game_over():
            action, payload = self.game_master.choose_action(self.shared_history)
            if action == "INJECT_ACTION":
                # Here you could further elaborate on how to handle game actions.
                self.shared_history.update_history(payload)
            elif action == "INJECT_CHARACTER_RESPONSE":
                char_name = payload["character"]
                guidance = payload.get("guidance", "")
                # It's assumed that guidance and shared_history are enough to generate a character's response.
                # Modify accordingly if more parameters are needed.
                response = self.character_gen.embody_character(self.characters[char_name], "tcc_context", guidance)
                self.shared_history.update_history({"Character": response})
            elif action == "WAIT_FOR_USER_INPUT":
                # Assume a getInput method that fetches user input
                user_input = self.getInput()
                self.shared_history.update_history({"Human": user_input})
            else:
                print(f"Invalid action: {action}")
        print("Game over.")

    def getInput(self):
        """
        Fetches user input.
        """
        return input("> ")


# %%
history = SharedHistory(20)

# %%
from langchain.memory import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage


class PlayerMessage(HumanMessage):
    """Type of message that is spoken by the human."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "player"


class NPCMessage(AIMessage):
    """Type of message that is spoken by the AI."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "npc"


class GameMasterMessage(SystemMessage):
    """Type of message that is a system message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "gamemaster"
    
class GameMessageHistory(ChatMessageHistory):

    def add_gamemaster_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(GameMasterMessage(content=message))

    def add_player_message(self, message: str) -> None:
        """Add a user message to the store"""
        self.add_message(PlayerMessage(content=message))

    def add_npc_message(self, message: str) -> None:
        """Add an AI message to the store"""
        self.add_message(NPCMessage(content=message))


# %%
from typing import Any
from langchain.schema import BaseMessage, get_buffer_string
from langchain.memory.chat_memory import BaseChatMemory

def get_buffer_string(
    messages: List[BaseMessage], human_prefix: str = "Player", ai_prefix: str = "NPC"
) -> str:
    """Get buffer string of messages."""
    string_messages = []
    for m in messages:
        if isinstance(m, PlayerMessage):
            role = human_prefix
        elif isinstance(m, NPCMessage):
            role = ai_prefix
        elif isinstance(m, GameMasterMessage):
            role = "GameMaster"
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}: {m.content}"
        string_messages.append(message)

    return "\n".join(string_messages)

class GameSessionBufferWindowMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Player"
    ai_prefix: str = "NPC"
    memory_key: str = "history"  #: :meta private:
    k: int = 5

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""

        buffer: Any = self.buffer[-self.k * 2 :] if self.k > 0 else []
        if not self.return_messages:
            buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: buffer}


# %%
chat_history = GameMessageHistory()

# %%
chat_history.add_gamemaster_message(message="You are in a tavern")

# %%
chat_history.add_npc_message(message="LOKI OF ASGARD: Hello, traveller")

# %%
chat_history.add_player_message(message="Hi, Loki")

# %%
game_memory = GameSessionBufferWindowMemory(k=50, )

# %%
game_room = LLMChain(
    memory=game_memory,
    llm=fast_llm,
    prompt=PromptTemplate.from_template("You are a game master. This is the history: {history}.\n This is the user's input:{human_input}"),
    verbose=True
)

# %%
game_room.run(PlayerMessage(content="I want to play a high fantasy game today"))

# %%
game_room.run([PlayerMessage(content="Yes. I want to play the story of a knight, from the very start")])

# %%



