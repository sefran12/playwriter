# %% [markdown]
# # Character object
# 
# Characteristics:
# 
# - Internal state
# - Ambitions
# - Teleology
# - Philosophy
# - Physical State
# - Long Term Memory
# - Short Term Memory
# - Internal Contradictions
# 
# A character is a temporal construction, and there are two kinds of interactions: On a given time slice, all these characteristics interact amongst them, and on the temporal dimension these characteristics influence their future counterparts.
# What is important is how this internal system affect the actions of the character. So, ultimately, the characterisation drives the responses of the character to their environment.

# %%
import os
import sys
from typing import Dict, List
from pydantic import BaseModel
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

strong_llm = ChatOpenAI(model="gpt-4", temperature=0.2)
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


# %%
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

class CharacterGeneration:
    def __init__(self, strong_llm):
        self.strong_llm = strong_llm

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

    def refine_character(self, character: Character, rounds: int) -> Character:
        for i in range(rounds):
            character_response = self.refine_chain.run(character)
            character = self.character_parser.parse(character_response)
        return character
    
    def get_embody_prompt(self, character: Character, tcc_context: str, scene_description: str) -> str:
        character_string = ""
        for characteristic, content in  character.dict().items():
            character_string += (str(characteristic) + ": " + str(content) + "\n")
        input = {'tcc_context': tcc_context, 'character_profile': character_string, 'scene_description': scene_description}
        embodied_character_template = self.embodier_template.format_messages(**input)
        return embodied_character_template[0].content

    def embody_character(self, character: Character, tcc_context: str, scene_description: str) -> LLMChain:
        embody_prompt = self.get_embody_prompt(character, tcc_context, scene_description)
        template = f"""{embody_prompt}

        {{history}}
        Human: {{human_input}}
        Character:"""
        prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
        chatgpt_chain = LLMChain(
            llm=self.strong_llm,
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=50),
        )
        return chatgpt_chain


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

character_description = "Maria: Alejandro's love interest, a beautiful and intelligent woman who is intrigued by Alejandro's passion."

# %%
tcc_context = """
TELEOLOGY: The play explores the paradox of power and mortality, and the ultimate sacrifice that comes with great responsibility. It seeks to convey the message that power is not an antidote to human frailty and death, and that the choices we make, especially those involving self-sacrifice, define our legacy.

CONTEXT: The play is set in a sprawling, futuristic cyberpunk city, a blend of advanced technology and arcane magic. The city is a living, breathing entity, a reflection of the soul of its inhabitants. At its heart lies an ancient temple housing the remains of the old emperor, and the city is guarded by titanic stone giants. The city is on the brink of a catastrophe that could either destroy it or the world outside.

CHARACTERS:
1. Kid: A frail boy with the power to command stone giants. He is innocent, curious, and brave.
2. Old Emperor: The deceased ruler of the city, whose spirit guides the Kid. He is wise and regretful.
3. Stone Giants: Ancient guardians of the city. They are silent, obedient, and powerful.
4. The City: A character in itself, reflecting the soul of the author. It is vibrant, chaotic, and mysterious.
5. The World: The entity outside the city. It is vast, unknown, and threatening.
6. Kid's Mother: A loving and protective figure. She is nurturing and fearful.
7. The Oracle: A mysterious figure who predicts the city's fate. She is cryptic, ominous, and enigmatic.
8. The Rebel Leader: A charismatic figure leading a rebellion against the city. He is passionate, ruthless, and persuasive.
9. The Scientist: A genius trying to understand the Kid's powers. She is intelligent, ambitious, and ethical.
10. The Priest: The spiritual leader of the city. He is devout, manipulative, and influential.

NARRATIVE THREADS:
1. Kid's DISCOVERY of his powers in the CITY helps attain TELEOLOGY by showing the paradox of power and mortality.
2. The Old Emperor's GUIDANCE to the Kid in the CITY helps attain TELEOLOGY by teaching the responsibility that comes with power.
3. The Stone Giants' OBEDIENCE to the Kid in the CITY helps attain TELEOLOGY by showing the potential and danger of absolute power.
4. The City's REACTION to the Kid's powers in the CITY helps attain TELEOLOGY by reflecting the author's inner turmoil.
5. The World's THREAT to the City in the CITY helps attain TELEOLOGY by presenting the choice between self and others.
6. Kid's Mother's PROTECTION of the Kid in the CITY helps attain TELEOLOGY by showing the human side of the divine.
7. The Oracle's PROPHECY about the city's fate in the CITY helps attain TELEOLOGY by foreshadowing the inevitable.
8. The Rebel Leader's REVOLT against the city in the CITY helps attain TELEOLOGY by challenging the status quo.
9. The Scientist's STUDY of the Kid's powers in the CITY helps attain TELEOLOGY by exploring the nature of power.
10. The Priest's MANIPULATION of the city's faith in the CITY helps attain TELEOLOGY by questioning the morality of power.
"""

character_description = "Kid: A frail boy with the power to command stone giants. He is innocent, curious, and brave."

# %%
maria_alejandra = CharacterGeneration(strong_llm)

# %%
description_1 = maria_alejandra.generate_character(tcc_context=tcc_context, character_description=character_description)

# %%
print(description_1)

# %%
embody_prompt = maria_alejandra.get_embody_prompt(description_1,
                                                 scene_description="We are in the top of an old skyscraper, you and I. I have been losing my mind, invaded and broken by the miriad cybernetic implants implanted in me",
                                                 tcc_context=tcc_context)

# %%
print(embody_prompt)

# %%
maria_alejandra_body = maria_alejandra.embody_character(
    character=description_1,
    scene_description="We are in the top of an old skyscraper, you and I. I have been losing my mind, invaded and broken by the miriad cybernetic implants implanted in me",
    tcc_context=tcc_context
)

# %%
maria_alejandra_body.predict(human_input="Hi, Unai")


# %%
maria_alejandra_body.predict(human_input="It is. I think... I cannot see")


# %%
maria_alejandra_body.predict(human_input="Don't worry about me. You know, I'll always survive. Tell me, what has been going on the city lately? They have been telling me that the Priest has declared the Emperor has come back again")


# %%
maria_alejandra_body.predict(human_input="What do you... mean? Don't worry. I don't care that much anymore. The world is beautiful, right? (looks at Unai) You and I will forever be human, right?")

# %%
maria_alejandra_body.predict(human_input="Haha, you've grown, Unai. I, Lelond of the Slums, remember when we started here. Will you come to visit me when I sleep? I feel tired (sits in a beam, the sunset slowly sets). Tell me a tale of your achievements, would you?")


# %%
maria_alejandra_body.predict(human_input="How so?")


# %%
maria_alejandra_body.predict(human_input="Hah, I envy you, my friend. As for me, I did survive. The surgeries made the tumors go away. But what I got were tumors in my mind. I can feel them right now, their tendrils gnawing at my mind. The slow hum of the bioelectricity in the implants. It's a cacophony I cannot understand (his gaze evidences infinite sadness, and a bit of madness). At least Mother didn't see me die, before she left for the Eternal...")

# %%
maria_alejandra_body.predict(human_input="Don't worry about me, Unai. I'll go to the Eternal soon. You have to go back. The world is waiting for you. I'll be here, OK? (looks at him with a smile. The smile causas strange contractions on his face, since half of the face is of immutable steel). Come back once in a while to tell me of your adventures, will you promise?")

# %%
maria_alejandra_body.predict(human_input="No, you have to go. Goodbye, Unai.")

# %%
memory_buffer = [{'interaction_number': i,
                  'character': row.__class__,
                  'message': row.content}
                for i, row in enumerate(maria_alejandra_body.memory.chat_memory.messages)]
memory_buffer

# %%
import langchain

memory_buffer = [{'interaction_number': 0,
  'character': langchain.schema.HumanMessage,
  'message': 'Hi, Unai'},
 {'interaction_number': 1,
  'character': langchain.schema.AIMessage,
  'message': "(looking out over the city, a soft glow from the neon lights illuminating his face) Hello, friend. It's a beautiful view, isn't it?"},
 {'interaction_number': 2,
  'character': langchain.schema.HumanMessage,
  'message': 'It is. I think... I cannot see'},
 {'interaction_number': 3,
  'character': langchain.schema.AIMessage,
  'message': "(Reaches out, gently touching the human's arm) I'm sorry... I can describe it to you, if you'd like. The city is alive tonight, a symphony of lights and sounds."},
 {'interaction_number': 4,
  'character': langchain.schema.HumanMessage,
  'message': "Don't worry about me. You know, I'll always survive. Tell me, what has been going on the city lately? They have been telling me that the Priest has declared the Emperor has come back again"},
 {'interaction_number': 5,
  'character': langchain.schema.AIMessage,
  'message': "(Takes a deep breath, looking out over the city) Yes, the Priest has been stirring up the city with his words. But the Emperor... he's not back. Not in the way the Priest claims. (Pauses, his gaze distant) He's here, in a way, guiding me. But he's not the one pulling the strings."},
 {'interaction_number': 6,
  'character': langchain.schema.HumanMessage,
  'message': "What do you... mean? Don't worry. I don't care that much anymore. The world is beautiful, right? (looks at Unai) You and I will forever be human, right?"},
 {'interaction_number': 7,
  'character': langchain.schema.AIMessage,
  'message': "(Chuckles softly, glancing at the human) Yes, we're human. And the world... (His gaze drifts to the horizon, where the city meets the unknown) It's vast and mysterious, full of beauty and danger. It's like us, I suppose. Full of contradictions."},
 {'interaction_number': 8,
  'character': langchain.schema.HumanMessage,
  'message': "Haha, you've grown, Unai. I, Lelond of the Slums, remember when we started here. Will you come to visit me when I sleep? I feel tired (sits in a beam, the sunset slowly sets). Tell me a tale of your achievements, would you?"},
 {'interaction_number': 9,
  'character': langchain.schema.AIMessage,
  'message': "(Smiles softly, sitting down next to Lelond) Of course, I'll visit. And a tale... (Pauses, thinking) Well, there was this one time, not too long ago. I found a new way to communicate with the stone giants. It was like... like understanding a language you didn't know you knew. It was... (His eyes light up, the memory fresh and vivid) It was incredible."},
 {'interaction_number': 10,
  'character': langchain.schema.HumanMessage,
  'message': 'How so?'},
 {'interaction_number': 11,
  'character': langchain.schema.AIMessage,
  'message': '(Smiles, his eyes distant as he recalls the memory) It was like a song, a melody that resonated within me. I could feel their thoughts, their emotions. It was... overwhelming, but in a good way. It made me realize how much more there is to learn, to understand. (Pauses, his gaze returning to Lelond) It was a small step, but it felt like a leap.'},
 {'interaction_number': 12,
  'character': langchain.schema.HumanMessage,
  'message': "Hah, I envy you, my friend. As for me, I did survive. The surgeries made the tumors go away. But what I got were tumors in my mind. I can feel them right now, their tendrils gnawing at my mind. The slow hum of the bioelectricity in the implants. It's a cacophony I cannot understand (his gaze evidences infinite sadness, and a bit of madness). At least Mother didn't see me die, before she left for the Eternal..."},
 {'interaction_number': 13,
  'character': langchain.schema.AIMessage,
  'message': "(Looks at Lelond, a mixture of sympathy and determination in his eyes) I'm sorry, Lelond. I wish... I wish there was something I could do. (Pauses, his gaze drifting to the city below) But you're strong, stronger than anyone I know. You've survived so much. And your mother... she'd be proud of you. I'm sure of it."},
 {'interaction_number': 14,
  'character': langchain.schema.HumanMessage,
  'message': "Don't worry about me, Unai. I'll go to the Eternal soon. You have to go back. The world is waiting for you. I'll be here, OK? (looks at him with a smile. The smile causas strange contractions on his face, since half of the face is of immutable steel). Come back once in a while to tell me of your adventures, will you promise?"},
 {'interaction_number': 15,
  'character': langchain.schema.AIMessage,
  'message': "(Looks at Lelond, a soft smile on his face) I promise, Lelond. I'll come back and tell you all about it. (Pauses, his gaze drifting back to the city) The world is waiting, yes. But it can wait a little longer. (Turns back to Lelond, his smile warm and genuine) For now, let's just enjoy the sunset, okay?"},
 {'interaction_number': 16,
  'character': langchain.schema.HumanMessage,
  'message': 'No, you have to go. Goodbye, Unai.'},
 {'interaction_number': 17,
  'character': langchain.schema.AIMessage,
  'message': "(Looks at Lelond, his eyes filled with a mixture of sadness and determination) Goodbye, Lelond. (Pauses, standing up and looking out over the city one last time) I'll be back. I promise. (Turns and walks away, his silhouette disappearing into the neon glow of the city)"}]

# %%
memory_string = ''
for interaction in memory_buffer:
    memory_string += str(interaction['character'].__name__) + ": " + str(interaction['message']) + '\n'

print(memory_string)

# %%



