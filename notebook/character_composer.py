# %%
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

strong_llm = ChatOpenAI(model="gpt-4-0613", temperature=0.3)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.3)

with open(file="prompts/generators/INITIAL_HISTORY_TCC_GENERATOR.txt", mode="r+") as f:
    INITIAL_HISTORY_TCC_GENERATOR = f.read()

# %%
seed_description = """
A magical cyberpunk city where a little kid, born in an abandoned skyscraper \
discovers he has inherited powers beyond imagination akin to those of the old \
dead emperor of the city that sleeps in an old temple in the middle of the city \
in an old Dome. His power, though, do not make him immortal, nor invulnerable. \
He just has command over titanic stone giants, and apart from that he is as frail \
as any other human. The play delves into the nature of power and their senselessness \
with respect to the inevitability of death. Should end with the kid, now an old man \
having to choose between destroying the city to save the world, or to destroy the world \
to save the city. And he will die either way. The city is a metaphor for the author's soul
and his body.
"""

# %%
seed_template = PromptTemplate(
    template=INITIAL_HISTORY_TCC_GENERATOR,
    input_variables=["seed_description"]
)
seed_chain = LLMChain(llm=strong_llm, prompt=seed_template)
seed_response = seed_chain.run(seed_description)

# %%
print(seed_response)

# %%
with open(file="prompts/parsers/CHARACTER_LIST_PARSER.txt", mode="r+") as f:
    CHARACTER_LIST_PARSER = f.read()

# %%
print(CHARACTER_LIST_PARSER)

# %%
character_parser_template = PromptTemplate(
    template=CHARACTER_LIST_PARSER,
    input_variables=["tcc_context"]
)
character_parser_chain = LLMChain(llm=fast_llm, prompt=character_parser_template)
character_parser_response = character_parser_chain.run(seed_response)

# %%
print(character_parser_response)

# %%
from pydantic import BaseModel, Field
from typing import List

class Character(BaseModel):
    Character: str = Field(description="name of the character")
    Description: str = Field(description="description of the character")

class Characters(BaseModel):
    characters: List[Character] = Field(description="list of characters and their descriptions")

# %%
from langchain.output_parsers import PydanticOutputParser

character_parser = PydanticOutputParser(pydantic_object=Characters)
character_parser_template = PromptTemplate(
    template="Parse the character information from the context.\n{format_instructions}\n{context}\n",
    input_variables=["context"],
    partial_variables={"format_instructions": character_parser.get_format_instructions()},
)
print(character_parser_template)

# %%
character_parser_chain = LLMChain(llm=fast_llm, prompt=character_parser_template)

# Assuming seed_response is the initial response you have
character_parser_response = character_parser_chain.run(seed_response)

# %%
character_dictionary = character_parser.parse(character_parser_response).dict()
print(character_dictionary)

# %%
for character in character_dictionary['characters']:
    print(character['Character'])
    print(character[""])

# %%
with open(file="prompts/generators/FIRST_PASS_CHARACTER_DESIGNER.txt", mode="r+") as f:
    FIRST_PASS_CHARACTER_DESIGNER = f.read()

# %%
character_description = str(character_dictionary['characters'][0])

# %%
character_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_DESIGNER,
    input_variables=["tcc_context", "character_description"]
)
character_chain = LLMChain(llm=strong_llm, prompt=character_template)
character_response = character_chain.run(
    dict(tcc_context=seed_description, 
         character_description=character_description)
    )

# %%
print(character_response)

# %%
character_description = str(character_dictionary['characters'][1])

# %%
character_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_DESIGNER,
    input_variables=["tcc_context", "character_description"]
)
character_chain = LLMChain(llm=strong_llm, prompt=character_template)
character_response = character_chain.run(
    dict(tcc_context=seed_description, 
         character_description=character_description)
    )

# %%
print(character_response)

# %%
character_description = str(character_dictionary['characters'][2])

# %%
character_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_DESIGNER,
    input_variables=["tcc_context", "character_description"]
)
character_chain = LLMChain(llm=strong_llm, prompt=character_template)
character_response = character_chain.run(
    dict(tcc_context=seed_description, 
         character_description=character_description)
    )

# %%
print(character_response)

# %%
character_description = str(character_dictionary['characters'][3])

# %%
character_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_DESIGNER,
    input_variables=["tcc_context", "character_description"]
)
character_chain = LLMChain(llm=strong_llm, prompt=character_template)
character_response = character_chain.run(
    dict(tcc_context=seed_description, 
         character_description=character_description)
    )

# %%
print(character_response)

# %%
character_description = str(character_dictionary['characters'][4])

# %%
character_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_DESIGNER,
    input_variables=["tcc_context", "character_description"]
)
character_chain = LLMChain(llm=strong_llm, prompt=character_template)
character_response = character_chain.run(
    dict(tcc_context=seed_description, 
         character_description=character_description)
    )

# %%
print(character_response)

# %%
character_description = str(character_dictionary['characters'][8])

# %%
character_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_DESIGNER,
    input_variables=["tcc_context", "character_description"]
)
character_chain = LLMChain(llm=strong_llm, prompt=character_template)
character_response = character_chain.run(
    dict(tcc_context=seed_description, 
         character_description=character_description)
    )

# %%
print(character_response)

# %%
character_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_DESIGNER,
    input_variables=["tcc_context", "character_description"]
)
character_chain = LLMChain(llm=strong_llm, prompt=character_template)

character_info = {}
for character in character_dictionary['characters']:
    char_name = character['Character']
    print("Processing character", char_name)
    character_response = character_chain.run(
        dict(tcc_context=seed_description, 
            character_description=character_description)
    )
    character_info[char_name] = {
        'name': char_name,
        'epoch': 1,
        'HPPTI': character_response
    }

# %% [markdown]
# # Character Enrichment

# %%
char_hppti = """
1. History: The Protagonist was born in an abandoned skyscraper in the heart of a magical cyberpunk city. His parents, unknown to him, left him to fend for himself in the harsh realities of the city. As a child, he discovered his unique ability to command titanic stone giants, a power that once belonged to the old dead emperor of the city. Despite his extraordinary abilities, he remained a solitary figure, living in the shadows of the city's towering structures. His life was marked by a constant struggle for survival, and the burden of his powers only added to his isolation and vulnerability. Over time, he grew to understand the nature of power and its insignificance in the face of death.

2. Physical Condition: The Protagonist is an old man, frail and weathered by the harsh realities of life. He stands at an average height, with a thin, almost skeletal frame that belies his immense power. His hair is white as snow, and his eyes, a piercing blue, hold a depth of wisdom and sorrow. His skin is marked with the wrinkles of time and hardship. Despite his frailty, he possesses an aura of authority and command, especially when he invokes his power over the stone giants. His health is deteriorating, a constant reminder of his mortality.

3. Philosophy: The Protagonist believes in the transience of power and the inevitability of death. He views his abilities not as a gift, but as a burden, a constant reminder of his own vulnerability and mortality. He values life and freedom above all else, and despises the corruption and greed that power often brings. He believes in the inherent goodness of humanity, despite the harsh realities of his world. His guiding principle is to use his power to protect and serve, rather than to dominate and control.

4. Teleology: The Protagonist's primary goal is to maintain the balance between his city and the world. He is driven by a sense of duty and responsibility, and is willing to sacrifice himself to protect the city and its inhabitants. His ultimate objective, however, is to find a way to use his power to save both the city and the world, even if it means his own death. He is motivated by a deep love for his city, and a desire to leave a lasting legacy of peace and harmony.

5. Internal Contradiction: The Protagonist's greatest contradiction lies in his struggle between his desire to protect the city and his duty to save the world. He is torn between his love for his city and his responsibility towards the world. His power, which he views as a burden, becomes a source of internal conflict, as he grapples with the decision to destroy either the city or the world. This contradiction adds depth to his character, and creates potential for internal conflict and growth.
"""

# %%
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper

# create tools
search = GoogleSerperAPIWrapper(K=5)
google_search_tool = Tool(
    name="Search",
    func=search.run,
    description="useful for when you need to answer questions about current events or non-encyclopedic matters"
)

wiki = WikipediaAPIWrapper(doc_content_chars_max=500)
wiki_tool = Tool(
    name="WikiSearch",
    func=lambda q: str(wiki.run(q)),
    description="useful to find information about encyclopedic matters",
    return_direct=True
)

# define tool roster
tools = [wiki_tool]


# %%
agent = initialize_agent([], strong_llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# %%
with open(file="prompts/generators/FIRST_PASS_CHARACTER_ENRICHMENT.txt", mode="r+") as f:
    FIRST_PASS_CHARACTER_ENRICHMENT = f.read()

# %%
writer_enrichment_template = PromptTemplate(
    template=FIRST_PASS_CHARACTER_ENRICHMENT,
    input_variables=["hppti_context"]
)
writer_enrichment_prompt = writer_enrichment_template.format(hppti_context=char_hppti)
print(writer_enrichment_prompt)

# %%
enrichment_response = agent.run(writer_enrichment_prompt)

# %%
enrichment_response = strong_llm.predict(writer_enrichment_prompt)

# %%
print(enrichment_response)

# %%



