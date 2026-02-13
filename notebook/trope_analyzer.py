# %%
import pandas as pd

# %%
trope_df = pd.read_csv("../data/TVTropesData/tropes.csv")
trope_df.info()

# %%
for trope in trope_df.Description[1:10]:
    print(trope)
    print("____")

# %%
trope_str = ''
for i, row in trope_df.sample(5).iterrows():
    trope_str += str(row.Trope) + ": " + str(row.Description) + '\n'

print(trope_str)

# %%
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

INITIAL_SCENE_TCC_GENERATOR = '''
You are part of a federation of artificial writers tasked with creating a play. \
Your role is to craft a succession of scenes based on four pieces of information given to you, \
and one piece of information you must come up with.

Given to you will be the following:

- a TELEOLOGY: This is the ultimate finality of the play, what corresponds to \
fate inside the story, or to the moral to a fable, or the ethical teaching to a parable. \
This teleology helps the different design decisions in the history be coherent amongst \
themselves as to push the narrative forward to its conclusion.
- a CONTEXT: This is a description of the long scale dealings surrounding the story. \
This may be thought as worldbuilding, and is the overall background where the play develops. \
Since this is a long scale view of the world, it will also evolve with the accumulation \
of scenes throughout the development of the story, so this is a description of its current state.
- a list of CHARACTERS: These are the actors that will populate the world of the play. \
These actors must be diverse and must express different sides of the TELEOLOGY. They must be \
generated as to enable the maximum contrast and dynamical potential in their pairwise interactions \
when these are made to interact in SCENES. This is a list with their names and a short description \
of their personalities, intentions and likeness.
- a list of NARRATIVE THREADS expressed as TROPES that use the CHARACTERS inside the CONTEXT \
that serve the overall TELEOLOGY of the play.

You must come up with:

- SCENE TROPES: These are the scene constructions that guide the interactions, like literary tropes.

You must use these pieces of information to come up with a list of scenes in the following format:

OVERALL TELEOLOGY: <please repeat the description of the teleology of the play, only once at the start, for downstream purposes>
SCENE NUMBER (X):
ACTORS: <actor name>, <actor name>, etc.
SETTING: <place where the scene takes place>
NARRATIVE THREADS: <list of narrative threads that participate in the scene>
LIST OF ACTIONS IN NARRATIVE ORDER:
- <actor 1> does <action>
- <action 1> talks with <actor 2> about <subject>
- etc...

And also a random list of TV and literary tropes will be given to you as "literary fate", a mechanism to enrich the narrative. You must use at least one of these in a creative way.

<<START OF CONTEXT>>
{tcc_context}
<<END OF CONTEXT>>

<<START OF TROPES LIST>>
{tropes_list}
<<END OF TROPES LIST>>


<<RESPONSE>>
'''

tcc_context = """TELEOLOGY: The play explores the paradox of power and mortality, and the ultimate sacrifice that comes with great responsibility. It seeks to convey the message that power is not an antidote to human frailty and death, and that the choices we make, especially those involving self-sacrifice, define our legacy.

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
10. The Priest's MANIPULATION of the city's faith in the CITY helps attain TELEOLOGY by questioning the morality of power."""

# %%
scene_template = PromptTemplate(
    template=INITIAL_SCENE_TCC_GENERATOR,
    input_variables=["tcc_context", "tropes_list"]
)
scene_prompt = scene_template.format(tcc_context=tcc_context, tropes_list=trope_str)
print(scene_prompt)

# %%
strong_llm = ChatOpenAI(model="gpt-4", temperature=0.2)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# %%
scene_chain = LLMChain(llm=strong_llm, prompt=scene_template)
scene_response = scene_chain.run({'tcc_context': tcc_context, 'tropes_list': trope_str})

# %%
print(scene_response)

# %%



