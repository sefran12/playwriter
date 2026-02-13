from dotenv import load_dotenv
from typing import Dict, List, Optional
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify

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
        with open(file="../../notebook/prompts/generators/FIRST_PASS_CHARACTER_DESIGNER.txt", mode="r+") as f:
            self.initial_character_generator = f.read()

        with open(file="../../notebook/prompts/refiners/FULL_DESCRIPTION_CHARACTER_REFINER.txt", mode="r+") as f:
            self.refine_character_generator = f.read()

        with open(file="../../notebook/prompts/embodiers/CHARACTER_EMBODIER.txt", mode="r+") as f:
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

# APP DEFINITION
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
character_gen = CharacterGeneration(strong_llm, fast_llm)

logging.basicConfig(filename='api.log', level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
app.logger.addHandler(console_handler)

@app.route('/generate', methods=['POST'])
def generate_character():
    data = request.get_json()
    tcc_context = data.get('tcc_context')
    character_description = data.get('character_description')
    character = character_gen.generate_character(tcc_context, character_description)
    app.logger.info(f"Generated character with tcc_context: {tcc_context} and character_description: {character_description}")
    return jsonify(character.dict())

@app.route('/chat', methods=['POST'])
def chat_with_character():
    data = request.get_json()
    character_data = data.get('character')
    tcc_context = data.get('tcc_context')
    scene_description = data.get('scene_description')
    human_input = data.get('human_input')

    character = Character(**character_data)
    chatgpt_chain = character_gen.embody_character(character, tcc_context, scene_description)
    response = chatgpt_chain.run({'history': '', 'human_input': human_input})
    app.logger.info(f"Chat with character: {character_data}, tcc_context: {tcc_context}, scene_description: {scene_description}, human_input: {human_input}")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)