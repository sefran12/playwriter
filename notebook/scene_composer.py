# %%
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

strong_llm = ChatOpenAI(model="gpt-4-0613", temperature=0.3)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.3)

with open(file="prompts/generators/FIRST_PASS_SCENE_DESIGNER.txt", mode="r+") as f:
    FIRST_PASS_SCENE_DESIGNER = f.read()

# %%
tcc_context = """TELEOLOGY: The ultimate finality of the play is to explore the nature of power, its futility in the face of mortality, and the moral dilemmas it presents. The story will end with the protagonist making a sacrificial choice between saving the city (symbolizing his soul) or the world (symbolizing his body), highlighting the transient nature of life and the enduring impact of our choices.

CONTEXT: The play is set in a magical cyberpunk city, a blend of advanced technology and arcane magic. The city is a sprawling metropolis with towering skyscrapers, neon-lit streets, and ancient temples. At the heart of the city lies an old Dome, where the old dead emperor of the city rests. The city is alive with diverse inhabitants, from tech-savvy hackers to mystical seers. The protagonist's powers over the stone giants have caused a shift in the city's dynamics, leading to power struggles, fear, and awe among the city's inhabitants.

CHARACTERS:
1. Protagonist: A frail boy with the power to control stone giants. He is wise beyond his years and struggles with his immense power and mortality.
2. Old Emperor: The deceased ruler of the city, whose powers the protagonist has inherited. He appears in dreams and visions.
3. Stone Giants: Titanic creatures of stone, controlled by the protagonist. They are silent but expressive.
4. Tech Hacker: A rebellious, tech-savvy inhabitant who initially opposes the protagonist but eventually becomes an ally.
5. Mystic Seer: A wise, old woman who guides the protagonist on his journey.
6. City Mayor: The political leader of the city, who fears the protagonist's power and seeks to control him.
7. Street Urchin: A street-smart, resourceful child who befriends the protagonist.
8. Cybernetic Enforcer: The Mayor's right hand, a ruthless enforcer with cybernetic enhancements.
9. Ancient Librarian: The keeper of the city's history and secrets, who aids the protagonist in understanding his powers.
10. Rebel Leader: The head of a resistance group against the city's current leadership, who sees the protagonist as a potential ally or threat.

NARRATIVE THREADS:
1. The Protagonist's struggle with his powers and mortality in the CONTEXT of the city helps attain TELEOLOGY by highlighting the futility of power against death.
2. The conflict between the Protagonist and the City Mayor in the CONTEXT of the city's political landscape serves TELEOLOGY by exploring the misuse of power.
3. The friendship between the Protagonist and the Street Urchin in the CONTEXT of the city's harsh realities serves TELEOLOGY by showing the power of human connection.
4. The guidance of the Mystic Seer to the Protagonist in the CONTEXT of the city's mystical heritage serves TELEOLOGY by exploring the wisdom needed to wield power responsibly.
5. The alliance between the Protagonist and the Tech Hacker in the CONTEXT of the city's technological landscape serves TELEOLOGY by showing the power of collaboration.
6. The opposition of the Cybernetic Enforcer to the Protagonist in the CONTEXT of the city's law enforcement serves TELEOLOGY by exploring the abuse of power.
7. The Protagonist's interaction with the Stone Giants in the CONTEXT of the city's ancient magic serves TELEOLOGY by showing the awe and fear power can inspire.
8. The Protagonist's learning from the Ancient Librarian in the CONTEXT of the city's history serves TELEOLOGY by highlighting the importance of understanding one's power.
9. The tension between the Protagonist and the Rebel Leader in the CONTEXT of the city's unrest serves TELEOLOGY by exploring the potential for power to be a tool or a weapon.
10. The Protagonist's final choice between the city and the world in the CONTEXT of his impending death serves TELEOLOGY by encapsulating the moral dilemmas power presents."""

# %%
class Place(BaseModel):
    name: str = Field(description="name of the place")
    description: str = Field(description="description of the place")

class Places(BaseModel):
    places: List[Place] = Field(description="list of places and their descriptions")

place_parser = PydanticOutputParser(pydantic_object=Places)
place_parser_template = PromptTemplate(
    template="Parse the place information from the context.\n{format_instructions}\n{context}\n",
    input_variables=["context"],
    partial_variables={"format_instructions": place_parser.get_format_instructions()},
)
place_parser_chain = LLMChain(llm=fast_llm, prompt=place_parser_template)



