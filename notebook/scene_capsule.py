# %%
import os
import sys
from dotenv import load_dotenv
from typing import Dict, List, Optional
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI

load_dotenv()

strong_llm = ChatOpenAI(model="gpt-4", temperature=0.2)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2)

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
    def __init__(self, strong_llm, fast_llm):
        self.strong_llm = strong_llm
        self.fast_llm = fast_llm

        # Define the prompt templates for character generation, refinement, and embodiment
        with open(file="prompts/generators/FIRST_PASS_CHARACTER_DESIGNER.txt", mode="r+") as f:
            self.initial_character_generator = f.read()

        with open(file="prompts/refiners/FULL_DESCRIPTION_CHARACTER_REFINER.txt", mode="r+") as f:
            self.refine_character_generator = f.read()

        with open(file="prompts/embodiers/CHARACTER_EMBODIER.txt", mode="r+") as f:
            self.character_embodier = f.read()

        self.character_parser = PydanticOutputParser(pydantic_object=Character)
        
        # Define the prompt chains for character generation, refinement, and embodiment
        self.init_template = PromptTemplate(
            template=self.initial_character_generator,
            input_variables=["tcc_context", "character_description"],
            partial_variables={"format_instructions": self.character_parser.get_format_instructions()},
        )
        self.init_chain = LLMChain(llm=self.strong_llm, prompt=self.init_template)

        self.refine_template = PromptTemplate(
            template=self.refine_character_generator,
            input_variables=["tcc_context", "character_profile"],
            partial_variables={"format_instructions": self.character_parser.get_format_instructions()},
        )
        self.refine_chain = LLMChain(llm=self.strong_llm, prompt=self.refine_template)

        self.embodier_template = ChatPromptTemplate.from_template(
            template=self.character_embodier
        )
        self.embodier_chain = LLMChain(llm=self.strong_llm, prompt=self.embodier_template)

    def generate_character(self, tcc_context:str, character_description: str) -> Character:
        input = {'tcc_context': tcc_context, 'character_description': character_description}
        character_response = self.init_chain.run(input)
        character = self.character_parser.parse(character_response)
        return character

    def refine_character(self, character: Character, tcc_context, rounds: int) -> Character:
        input = {'tcc_context': tcc_context, 'character_profile': character}
        for i in range(rounds):
            character_response = self.refine_chain.run(input)
            character = self.character_parser.parse(character_response)
        return character
    
    def get_embody_prompt(self, character: Character, tcc_context: str, scene_description: str) -> str:
        character_string = ""
        for characteristic, content in  character.dict().items():
            character_string += (str(characteristic) + ": " + str(content) + "\n")
        input = {'tcc_context': tcc_context, 'character_profile': character_string, 'scene_description': scene_description}
        embodied_character_template = self.embodier_template.format_messages(**input)
        return embodied_character_template[0].content

    def embody_character(self, character: Character, tcc_context: str, scene_description: str, strong: bool = True, verbose: bool = False) -> LLMChain:
        if strong:
            llm = self.strong_llm
        else:
            llm = self.fast_llm
        embody_prompt = self.get_embody_prompt(character, tcc_context, scene_description)
        template = f"""{embody_prompt}

        {{history}}
        Input: {{human_input}}
        Character:"""
        prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
        chatgpt_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=verbose,
            memory=ConversationBufferWindowMemory(k=50),
        )
        return chatgpt_chain
    
class CharacterBody:
    def __init__(self, chain: LLMChain, name: Optional[str] = None):
        self.name = name
        self.chain = chain

    def forget(self):
        self.memory.chat_memory.clear()

    def set_name(self, name: str):
        self.name = name

    def set_prompt(self, prompt: PromptTemplate):
        self.chain.prompt = prompt

    def predict(self, message: str):
        return self.chain.predict(human_input=message)


# %%
class Scene:
    def __init__(self, character1_embodiment: LLMChain, character2_embodiment: LLMChain):
        self.character1_chain = character1_embodiment
        self.character2_chain = character2_embodiment
        self.history = []

    def converse(self, start_character: str, max_rounds: int = 20):
        current_character = start_character
        previous_output = ''
        round = 0
        while round < max_rounds:
            round += 1
            if current_character == "character1":
                current_chain = self.character1_chain
            else:
                current_chain = self.character2_chain

            current_message = current_chain.predict(human_input=previous_output)
            print(f"{current_character}: {current_message}")
            previous_output = current_message
            current_character = "character1" if current_character == "character2" else "character2"

# %%
tcc_context = """TELEOLOGY: The play will ultimately demonstrate the futility of grandiose ambition and the importance of appreciating the simple, everyday aspects of life. The tragicomic ending will serve as a reminder of the unpredictability and irony of life, and the inherent humor in human attempts to understand and control it.

CONTEXT: The play is set in contemporary Lima, a city of contrasts, where the old and the new coexist, where poverty and wealth live side by side, and where dreams and reality often collide. The city itself is a character, with its vibrant street life, its bustling markets, its dangerous slums, and its tranquil parks. The protagonist's philosophical quest takes him through all these aspects of the city, providing a rich backdrop for the story.

CHARACTERS:
1. Alejandro: The protagonist, a young man with grand ambitions of revolutionizing philosophy.
2. Rosa: Alejandro's mother, a practical woman who worries about her son's impractical dreams.
3. Carlos: Alejandro's best friend, a street-smart guy who provides a contrast to Alejandro's intellectual pursuits.
4. Professor Mendoza: Alejandro's philosophy professor, a cynical man who doesn't believe in Alejandro's ideas.
5. Maria: Alejandro's love interest, a beautiful and intelligent woman who is intrigued by Alejandro's passion.
6. Don Julio: A wealthy businessman who becomes interested in Alejandro's ideas.
7. Lucia: Don Julio's daughter, a spoiled and shallow woman who provides a contrast to Maria.
8. Pedro: A street vendor who becomes Alejandro's unlikely friend and philosophical sounding board.
9. Inspector Gomez: A police officer who becomes involved in Alejandro's life after a minor incident.
10. The Combi Driver: A minor character who plays a major role in the story's tragicomic ending.

NARRATIVE THREADS:
1. Alejandro's QUEST FOR KNOWLEDGE in the CONTEXT of Lima's diverse society helps attain the TELEOLOGY by showing the contrast between his lofty ambitions and the harsh realities of life.
2. Rosa's WORRY FOR HER SON in the CONTEXT of their modest home life helps attain the TELEOLOGY by highlighting the practical concerns that Alejandro ignores in his pursuit of philosophy.
3. Carlos's STREET SMARTS in the CONTEXT of Lima's dangerous streets helps attain the TELEOLOGY by providing a contrast to Alejandro's intellectual pursuits.
4. Professor Mendoza's SKEPTICISM in the CONTEXT of the academic world helps attain the TELEOLOGY by challenging Alejandro's ideas and forcing him to defend them.
5. Maria's INTEREST IN ALEJANDRO in the CONTEXT of their budding romance helps attain the TELEOLOGY by showing a softer, more human side of Alejandro.
6. Don Julio's INTEREST IN ALEJANDRO'S IDEAS in the CONTEXT of his business empire helps attain the TELEOLOGY by showing the potential commercialization of philosophy.
7. Lucia's SHALLOWNESS in the CONTEXT of Lima's wealthy society helps attain the TELEOLOGY by providing a contrast to Maria's depth and intelligence.
8. Pedro's FRIENDSHIP WITH ALEJANDRO in the CONTEXT of their unlikely pairing helps attain the TELEOLOGY by showing that wisdom can come from unexpected places.
9. Inspector Gomez's INVOLVEMENT IN ALEJANDRO'S LIFE in the CONTEXT of a minor incident helps attain the TELEOLOGY by introducing an element of danger and unpredictability.
10. The Combi Driver's ROLE IN ALEJANDRO'S DEATH in the CONTEXT of Lima's chaotic traffic helps attain the TELEOLOGY by providing the tragicomic ending that underscores the unpredictability and irony of life."""

character_description_1 = "Alejandro: The protagonist, a young man with grand ambitions of revolutionizing philosophy."
character_description_2 = "Maria: Alejandro's love interest, a beautiful and intelligent woman who is intrigued by Alejandro's passion."

# %%
char_generator = CharacterGeneration(strong_llm, fast_llm)

# %%
alejandro = char_generator.generate_character(tcc_context=tcc_context, character_description=character_description_1)
maria = char_generator.generate_character(tcc_context=tcc_context, character_description=character_description_2)

# %%
alejandro2 = char_generator.refine_character(tcc_context, alejandro, 1)

# %%
for char1, char2 in zip(alejandro.dict(), alejandro2.dict()):
    print(char1, ":")
    print("before refinement:", alejandro.dict()[char1])
    print("after refinement:", alejandro2.dict()[char2])
    print("___")

# %%
scene_description = "The Parque del Amor, in Miraflores, Lima. Sunset in a slow April. Warm. Love is in the air"

# %%
alejandro_body = char_generator.embody_character(
    character=alejandro,
    scene_description=scene_description,
    tcc_context=tcc_context,
    strong=True,
    verbose=False
)
maria_body = char_generator.embody_character(
    character=maria,
    scene_description=scene_description,
    tcc_context=tcc_context,
    strong=True,
    verbose=False
)

# %%
alejandro_prompt = char_generator.get_embody_prompt(alejandro, tcc_context, scene_description)

# %%
print(alejandro_prompt)

# %%
#alejandro_body = alejandro_body.chain
#maria_body = maria_body.chain

# %%
#alejandro_body = CharacterBody(alejandro_body, "Alejandro")
#maria_body = CharacterBody(maria_body, "Maria")

# %%
alejandro_body.memory.chat_memory.clear()
maria_body.memory.chat_memory.clear()

# %%
maria_body.memory.chat_memory

# %%
scene = Scene(alejandro_body, maria_body)

# %%
scene.converse("character1", max_rounds=3)

# %%
maria_body.memory.chat_memory.dict()

# %%
alejandro_body.memory.chat_memory.dict()

# %%



