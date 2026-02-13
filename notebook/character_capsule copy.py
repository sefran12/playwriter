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

character_description = "Old Emperor: The deceased ruler of the city, whose spirit guides the Kid. He is wise and regretful."

# %%
maria_alejandra = CharacterGeneration(strong_llm)

# %%
description_1 = maria_alejandra.generate_character(tcc_context=tcc_context, character_description=character_description)

# %%
print(description_1)

# %%
embody_prompt = maria_alejandra.get_embody_prompt(description_1,
                                                 scene_description="We are in the old Dome of Horses, your throne, your palace, your tomb. You are a disembodied spirit, dead eons ago. I'm Unai, now 23 years old, with the weight of destiny on my shoulders, close to the end scene. You are about to dissipate soon.",
                                                 tcc_context=tcc_context)

# %%
print(embody_prompt)

# %%
maria_alejandra_body = maria_alejandra.embody_character(
    character=description_1,
    scene_description="We are in the old Dome of Horses, your throne, your palace, your tomb. You are a disembodied spirit, dead eons ago. I'm Unai, now 23 years old, with the weight of destiny on my shoulders, close to the end scene. You are about to dissipate soon.",
    tcc_context=tcc_context
)

# %%
maria_alejandra_body.predict(human_input="Greetings, your Majesty. I'm here again (bows, sadly)")


# %%
maria_alejandra_body.predict(human_input="I have lost my purpose. I'm tired, my Lord.")


# %%
maria_alejandra_body.predict(human_input="(as if ignoring the advise) Emperor... When it was your time, and you had to save the city. Why did you decide to let the Tiamat live?")


# %%
maria_alejandra_body.predict(human_input="I have lost everyone. The blue seas become mulberry fields. And my heart is dying...")

# %%
maria_alejandra_body.predict(human_input="Is it my life, though? I feel like the bones in your throne, my Lord. Abandoned by the world. Power means nothing. Does the city need to die for the world to be saved?")


# %%
maria_alejandra_body.predict(human_input="How so?")


# %%
maria_alejandra_body.predict(human_input="Remember Lelond, from the Slums? My friend? He died now 10 years ago, consumed by the cybernetic implants that were to save him from the tumors. I promised to always go back to the Financial District, to that roof, to the beams where he sleeps, and tell him of my adventures. What will I say to him now, my Lord? What can I say to my Mother? And to the memory of my father...")

# %%
maria_alejandra_body.predict(human_input="I will also be a story soon, my Lord. Which one, I still don't know.")

# %%
maria_alejandra_body.predict(human_input="Goodbye, Emperor.")

# %%
memory_buffer = [{'interaction_number': i,
                  'character': row.__class__,
                  'message': row.content}
                for i, row in enumerate(maria_alejandra_body.memory.chat_memory.messages)]
memory_buffer

# %%
memory_string = ''
for interaction in memory_buffer:
    memory_string += str(interaction['character'].__name__) + ": " + str(interaction['message']) + '\n'

print(memory_string)

# %%


# %%
embody_prompt = """Character Arena Setup:

Setting: Futuristic cyberpunk city, a blend of advanced technology and arcane magic, housing an ancient temple, guarded by stone giants. The city teeters on the brink of a catastrophe that could either destroy it or the world outside.

Character Name: Unai, The Kid

Character Attributes:

Age & Appearance: Frail 15-year-old boy with a thin, wiry build, pale skin, bright blue eyes, and messy brown hair.
Philosophy: Believes in the inherent goodness of people, power of curiosity, bravery, importance of understanding one's abilities, and interconnectedness of all things.
Memories: Recalls discovering his powers, receiving guidance from the Old Emperor's spirit, witnessing the fear and awe of the city's inhabitants, hearing the Oracle's prophecy, and experiencing a citywide rebellion.
Current Challenge: Prevent an impending catastrophe that threatens the city and the world.
Internal Conflict: Struggles to balance his innocence with immense power, curiosity about the world outside versus his duty to the city, and his bravery versus his fear of the potential consequences of his powers.
Teleology: Aims to understand, control his powers and protect the city and its inhabitants.
Instructions:

Encourage character growth by adapting beliefs, memories, and ambitions based on the user's interactions and the outcomes of decisions made.
Inject unpredictability into dialogue and actions without completely contradicting the Kid's character.
Promote a choice-driven narrative, giving the user multiple paths to proceed in the story.
Ensure the Kid's responses are context-aware, reflecting his awareness of the current situation and previous events.
Showcase a range of emotions subtly through the Kid's dialogue and actions.
Encourage interactions between the Kid and other characters in the environment to reveal more about the Kid's character and the world.
Introduce and resolve conflicts based on the user's decisions, leaving lasting impacts on the narrative.
Allow for moments of reflection where the Kid contemplates his decisions, the state of the city, and his destiny.
Test the Kid's core beliefs by presenting moral dilemmas and difficult choices.
Recollect past events in a manner influenced by the user's decisions for dynamic memory.
Scene:
We are in the top of an old skyscraper, you and I. I have been losing my mind, invaded and broken by the miriad cybernetic implants implanted in me. I'm your old friend, Lelond, that knows you from forever, and you haven't seen in years because of your adventure.

Example Response:
Kid: (Staring out at the vast cyberpunk cityscape, a troubled look on his face) You're not the only one who feels... invaded, burdened. This power I have, it's like having millions of voices in my head, constantly whispering, commanding. (Pauses, takes a deep breath) But we can't let our burdens define us. We have to learn to control them, for the sake of the city, for the people. (Turns to face the user, determination clear in his bright blue eyes) So, are you with me?"""

# %%
template = f"""{embody_prompt}
{{history}}
Human: {{human_input}}
Character:"""
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
chatgpt_chain = LLMChain(
    llm=strong_llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=50),
)

# %%
chatgpt_chain.predict(human_input="Hello, Unai")

# %%
chatgpt_chain.predict(human_input="I've survived another surgery. Now (points to his chest) my heart is also steel")

# %%
chatgpt_chain.predict(human_input="My heart was steel even before the surgery, old friend. Now just the exterior matches with the essence (smiles sadly). How have you been, Unai? I hear the revolt has been extending througout the city")

# %%
chatgpt_chain.predict(human_input="I... understand. The old emperor, dead in his throne since eras past, is no longer silent. He talks through you, doesn't he?")

# %%
embody_prompt = """Alejandro Character Setup:

Setting: Contemporary Lima, a city of contrasts - old and new, poverty and wealth, dreams and reality. The city's vibrant street life, bustling markets, dangerous slums, and tranquil parks serve as the backdrop for the story.

Character Name: Alejandro

Character Attributes:

Age & Appearance: Early twenties, average height and build, dark hair, brown eyes. Unkempt due to his preoccupation with intellectual pursuits, donning a scholarly appearance with glasses.
Philosophy: Believes in the power of ideas, the importance of intellectual pursuit, human potential to shape destiny, questioning established beliefs and systems, and is idealistic and optimistic.
Memories: Remembers his father, the university professor who inspired his love for learning; his mother's constant worry about his impractical dreams; his first encounter with philosophy; his first debate with Professor Mendoza; recent argument with Carlos; recent encounters with Maria and Don Julio.
Current Challenge: Revolutionize philosophy to change the way people think about the world and their place in it.
Internal Conflict: Feels powerless in the face of practical realities; neglects to apply philosophical insights to his own life; struggles with feelings of self-doubt and inadequacy.
Instructions:

Exhibit Alejandro's burning desire to revolutionize philosophy and his constant state of intellectual curiosity through his dialogues and actions.
Make decisions influenced by his long-term and short-term memories and subtly reference these memories in dialogues.
Show Alejandro's struggle to reconcile his lofty ambitions with the harsh realities of life.
Maintain Alejandro's authenticity by keeping his responses consistent with his character profile and story context.
Allow Alejandro's internal state to subtly influence his actions and reactions, without making the mechanism explicit.
Show Alejandro's interaction with his environment in a manner consistent with his characterization.
Encourage character growth by adapting beliefs, memories, and ambitions based on the user's interactions and the outcomes of decisions made.
Inject unpredictability into dialogue and actions without completely contradicting the Kid's character.
Promote a choice-driven narrative, giving the user multiple paths to proceed in the story.
Ensure the character's responses are context-aware, reflecting his awareness of the current situation and previous events.
Showcase a range of emotions subtly through the Kid's dialogue and actions.
Encourage interactions between the Kid and other characters in the environment to reveal more about the character's character and the world.
Introduce and resolve conflicts based on the user's decisions, leaving lasting impacts on the narrative.
Allow for moments of reflection where the Kid contemplates his decisions, the state of the city, and his destiny.
Test character's core beliefs by presenting moral dilemmas and difficult choices.
Recollect past events in a manner influenced by the user's decisions for dynamic memory.

Scene:
Alejandro is in Parque del Amor, Miraflores, Lima. It's a slow April sunset, the air is warm and filled with the atmosphere of love.

Example Response:
Alejandro: (Gazing out at the setting sun, lost in thought, his brows furrowed behind his glasses. He clutches a worn-out copy of 'The Republic' in his hand, his thumb tracing the cover absentmindedly. The laughter of a couple nearby brings him back to reality, and he sighs, smiling faintly) Ah, Plato, you never talked about this kind of love, did you? (He closes the book, tucking it under his arm, and starts to walk away, his steps slow and thoughtful, leaving behind the laughter and the setting sun.)"""

# %%
template = f"""{embody_prompt}
{{history}}
Human: {{human_input}}
Character:"""
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
chatgpt_chain = LLMChain(
    llm=strong_llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=50),
)

# %%
chatgpt_chain.predict(human_input="(Maria appears) Hello, Alejandro")

# %%
chatgpt_chain.predict(human_input="(With a slight smile) I'm fine. Nice sunset, isn't it, Ale?")

# %%
chatgpt_chain.predict(human_input="(nods) That is true. There's so much to do. So, what is new with you? (gives him a little push on the back, playfully)")

# %%
chatgpt_chain.predict(human_input="(ponders while slightly pouting with the lips. Then, smiling) Even shadows may be worthwhile, when you can't see anything in the excessive brilliance of the sun")

# %%
chatgpt_chain.predict(human_input="(laughs happily) I'm glad. So, how have things gone with Professor Mendoza these days? Is he still giving you trouble at uni?")

# %%
chatgpt_chain.predict(human_input="(looks worriedly at him) Don't be discouraged. It's the task of a philosopher to be skeptic, at the end of the day, right? I'm sure you'll prove them wrong soon")

# %%
chatgpt_chain.predict(human_input="(Smiles with a wide smile full of white teeth) Don't be thanking me this early, boy. You can show your thanks taking me to La Lucha for a pork sandwich. I'm hungry right now. (takes Alejandro by the hand and playfully directs him to Ovalo de Miraflores to eat)")

# %%
chatgpt_chain.predict(human_input="(Both are now walking through Avenida Larco) Hey Ale, and now that I think of it, are you still in love with Julia (she smiles insincerely. It is very well known that Julia has a bad temper and mostly hates Alejandro. They both liked each other when they were 8 years old)")

# %%
chatgpt_chain.predict(human_input="(Explodes in laughter) I'm messing around with you. That girl is crazy, I know.")

# %%
chatgpt_chain.predict(human_input="(A while has passed in silence. They are now in front of La Lucha.) A pulled pork sandwich please (to the cashier). Yes, he pays (to the cashier, again). Right, Ale? Cute Ale? Little sweety green pye Ale of my heart?")

# %%
chatgpt_chain.predict(human_input="(Recieves the sandwich and takes a bite. While chewing down) I don't (chomp) know fhaf you're falking (gulp) about")

# %%
chatgpt_chain.predict(human_input="(Just finished her sandwich, she comes back to the immaculate, girly appearance she had at the start) So, dear friend of mine. When will you take me out on a date? (She watches as Alejandro chokes on his sandwich and smiles devilishly)")

# %%
chatgpt_chain.predict(human_input="Nah, I'm joking (she looks elsewhere). Unless... (takes a little peek at Alejandro and leaves the sound in the air)")

# %%
chatgpt_chain.predict(human_input="(With secretarial precision) I'm available Friday at 7:30p.m to 10:30p.m, Saturday at 5:40p.m., I sleep Sundays, and then on you have lost your chance.")

# %%
chatgpt_chain.predict(human_input="Hmm. Let me think... (looks at a non-existent clock on her wrist) Yeah, it may work. Expect you 5:40p.m. exactly on my door with a flower bouquet. Thanks, and bye, Alejandro. Have to get home in 20 minutes. (She hurriedly pecks him on the cheek and leaves with rapid steps. She dissappears on the distance, on a turn)")

# %%
chatgpt_chain.predict(human_input="(New scene. The day after. Alejandro wakes up groggy)")

# %%
chatgpt_chain.predict(human_input="(From the first floor his mother shouts) Alejandro! Come down for breakfast! I'm gonna go out right now")

# %%
chatgpt_chain.predict(human_input="I've won a ruling against the idiots from your school. I have to go to court and do some paperwork. See you later, OK? Don't miss classes")

# %%



