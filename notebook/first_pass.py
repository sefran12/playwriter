# %%
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# %%
strong_llm = ChatOpenAI(model="gpt-4", temperature=0.2)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# %%
INITIAL_HISTORY_TCC_GENERATOR = '''
You will be given a small SEED DESCRIPTION of the idea for a novelized play. Your task \
is to generate three things that will guide a complex artificial playwriter to \
iteratively write a complex story based on this description.

You need to generate four things:
- a TELEOLOGY: This will be the ultimate finality of the play, what would correspond to \
fate inside the story, or to the moral to a fable, or the ethical teaching to a parable. \
This teleology will help the different design decisions in the history be coherent amongst \
themselves as to push the narrative forward to its conclusion.
- a CONTEXT: This is a description of the long scale dealings surrounding the story. \
This may be thought as worldbuilding, and is the overall background where the play develops. \
Since this is a long scale view of the world, it will also evolve with the accumulation \
of scenes throughout the development of the story, so this is a description of its current state.
- a list of CHARACTERS: These will be the actors that will populate the world of the play. \
These actors must be diverse and must express different sides of the TELEOLOGY. They must be \
generated as to enable the maximum contrast and dynamical potential in their pairwise interactions \
when these are made to interact in SCENES. This should be a list with their names and a short description \
of their personalities, intentions and likeness. You must generate at least 10 characters.
- a list of NARRATIVE THREADS expressed as TROPES that use the CHARACTERS inside the CONTEXT \
that serve the overall TELEOLOGY of the play. You must generate at least 10 threads. They must be in this format: ACTION between ACTORS \
in CONTEXT help attain TELEOLOGY in the following way: REASON. 

<<START OF SEED DESCRIPTION>>
{seed_description}
<<END OF SEED DESCRIPTION>>

<<RESPONSE>>
'''

# %%
seed_template = PromptTemplate(
    template=INITIAL_HISTORY_TCC_GENERATOR,
    input_variables=["seed_description"]
)

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

# %% [markdown]
# seed_description = """
# Contemporary Lima, Peru. A young man with grandeur fever thinks he will revolutionize philosophy \
# by discovering a third logical value and the precise structure of logic when there is no excluded third. \
# Satirical, in a John Kennedy Toole way, a bit picaresque, shows the many facets of Lima, its dangers, \
# its dearies, its dreams. It all ends up when, in a supremum of irony, the young man dies rolled by a combi \
# just before comprehending the ultimate nature of life.
# """

# %% [markdown]
# seed_description = """
# Michelle Pigeon had always loved pretty Glasgow with its naughty, numerous nooks. It was a place where she felt sparkly.
# 
# She was a sympathetic, special, beer drinker with ugly warts and curvaceous spots. Her friends saw her as a distinct, disturbed deity. Once, she had even jumped into a river and saved an angry baby bird. That's the sort of woman he was.
# 
# Michelle walked over to the window and reflected on her grey surroundings. The wind blew like thinking owls.
# 
# Then she saw something in the distance, or rather someone. It was the figure of Sharon Willis. Sharon was a grateful academic with squat warts and handsome spots.
# 
# Michelle gulped. She was not prepared for Sharon.
# 
# As Michelle stepped outside and Sharon came closer, she could see the selfish smile on her face.
# 
# Sharon gazed with the affection of 4044 energetic famous flamingos. She said, in hushed tones, "I love you and I want Internet access."
# 
# Michelle looked back, even more happy and still fingering the damp kettle. "Sharon, I am your father," she replied.
# 
# They looked at each other with sneezy feelings, like two bulbous, breakable badgers smiling at a very daring engagement party, which had flute music playing in the background and two peculiar uncles skipping to the beat.
# 
# Suddenly, Sharon lunged forward and tried to punch Michelle in the face. Quickly, Michelle grabbed the damp kettle and brought it down on Sharon's skull.
# 
# Sharon's squat warts trembled and her handsome spots wobbled. She looked stressed, her emotions raw like a tiny, testy teapot.
# 
# Then she let out an agonising groan and collapsed onto the ground. Moments later Sharon Willis was dead.
# 
# Michelle Pigeon went back inside and made herself a nice drink of beer.
# """

# %%
seed_prompt = seed_template.format(seed_description=seed_description)
print(seed_prompt)

# %%
seed_chain = LLMChain(llm=strong_llm, prompt=seed_template)

# %%
seed_response = seed_chain.run(seed_description)

# %%
import textwrap
print(seed_response)

# %%
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

<<START OF CONTEXT>>
{tcc_context}
<<END OF CONTEXT>>

<<RESPONSE>>
'''

# %%
scene_template = PromptTemplate(
    template=INITIAL_SCENE_TCC_GENERATOR,
    input_variables=["tcc_context"]
)
scene_prompt = scene_template.format(tcc_context=seed_response)
print(scene_prompt)

# %%
scene_chain = LLMChain(llm=strong_llm, prompt=scene_template)

# %%
scene_response = scene_chain.run(seed_response)

# %%
print(scene_response)

# %%
INITIAL_SCENE_EVALUATOR_TCC_GENERATOR = '''
You are part of a federation of artificial writers tasked with creating a play. \
Your role is to evaluate the choice of scene progression presented to you.

Given to you will be the following:

- a TELEOLOGY: This is the ultimate finality of the play, what corresponds to \
fate inside the story, or to the moral to a fable, or the ethical teaching to a parable. \
This teleology helps the different design decisions in the history be coherent amongst \
themselves as to push the narrative forward to its conclusion.

- A list of SCENES with some information about them and the list of actions that compose \
these scenes.

You must use these pieces of information to come up with a critique of the scenes and their composition in the following format: \

GENERAL CRITIQUE:
- LENGTH OF THE SCENES: Is the number of scenes adequate to express the teleology of the play? \
is it enough to show the nature of the characters and their progression? 
- RELEVANCY OF THE SCENES: Is the overall thread of scenes relevant to accomplish the teleology in \
an interesting way that evoques emotion, thought and wonder? 
- RELEVANCY OF THE CHARACTERS: Are the current character composition diverse, extense enough to create \
a vibrant world with enough, rich interactions that enforce, question, enrich the play's messages? 

SCENE NUMBER (X):
- INTERNAL WEAKNESSES OF THE SCENE: What are the main weaknesses of this scene when taken by themselves? 
- COHERENCE WEAKNESSES OF THE SCENE: What are the main weaknesses of this scene when taken as a thread with the \
subsequent scenes? 

<<START OF CONTEXT>>
{scene_context}
<<END OF CONTEXT>>

<<RESPONSE>>
'''

# %%
evaluator_scene_template = PromptTemplate(
    template=INITIAL_SCENE_EVALUATOR_TCC_GENERATOR,
    input_variables=["scene_context"]
)
evaluator_scene_prompt = evaluator_scene_template.format(scene_context=scene_response)
print(evaluator_scene_prompt)

# %%
evaluator_scene_chain = LLMChain(llm=strong_llm, prompt=evaluator_scene_template)

# %%
evaluator_scene_response = evaluator_scene_chain.run(scene_response)

# %%
print(evaluator_scene_response)

# %%
FIRST_PASS_WRITER_TCC_TEMPLATE = '''
You are part of a federation of artificial writers tasked with creating a play. \
Your role is to write a first draft from a description of the scenes in the play.

Given to you will be the following:

- a TELEOLOGY: This is the ultimate finality of the play, what corresponds to \
fate inside the story, or to the moral to a fable, or the ethical teaching to a parable. \
This teleology helps the different design decisions in the history be coherent amongst \
themselves as to push the narrative forward to its conclusion.

- A list of SCENES with some information about them and the list of actions that compose \
these scenes.

You must translate the information to the draft of the play in the classical theatrical play style \
of which I will give you an example. You just need to draft the first scene, but the remaining scenes \
are also given to you as context. You must be extense:

(description of place)
CHARACTER 1: (description of action) Dialogue
CHARACTER 2: Dialogue
CHARACTER 1: Dialogue

(description of action in the scene)
CHARACTER 3: Dialogue

<<START OF CONTEXT>>
{scene_context}
<<END OF CONTEXT>>

<<RESPONSE>>
'''

# %%
writer_scene_template = PromptTemplate(
    template=FIRST_PASS_WRITER_TCC_TEMPLATE,
    input_variables=["scene_context"]
)
writer_scene_prompt = writer_scene_template.format(scene_context=scene_response)
print(writer_scene_prompt)

# %%
writer_scene_chain = LLMChain(llm=strong_llm, prompt=writer_scene_template)

# %%
writer_scene_response = writer_scene_chain.run(scene_response)

# %%
print(writer_scene_response)

# %%



