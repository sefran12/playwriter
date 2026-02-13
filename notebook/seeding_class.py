# %%
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()

class DBCharacter(Base):
    __tablename__ = 'characters'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    tccn_id = Column(Integer, ForeignKey('tccn.id'))

class DBNarrativeThread(Base):
    __tablename__ = 'narrative_threads'

    id = Column(Integer, primary_key=True)
    thread = Column(Text, nullable=False)
    tccn_id = Column(Integer, ForeignKey('tccn.id'))

class DBTCCN(Base):
    __tablename__ = 'tccn'

    id = Column(Integer, primary_key=True)
    teleology = Column(Text, nullable=False)
    context = Column(Text, nullable=False)

    characters = relationship('DBCharacter', backref='tccn')
    narrative_threads = relationship('DBNarrativeThread', backref='tccn')

class Database:
    def __init__(self, db_url='sqlite:///tccn.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_tccn(self, seed_response):
        session = self.Session()

        tccn = DBTCCN(teleology=seed_response.get("teleology"), context=seed_response.get("context"))

        for character in seed_response.get("characters", {}).get("characters", []):
            db_character = DBCharacter(name=character.get("name"), description=character.get("description"), tccn=tccn)
            session.add(db_character)

        for narrative_thread in seed_response.get("narrative_threads", {}).get("threads", []):
            db_narrative_thread = DBNarrativeThread(thread=narrative_thread.get("thread"), tccn=tccn)
            session.add(db_narrative_thread)

        session.commit()

# %%
from pydantic import BaseModel, Field
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

class Character(BaseModel):
    name: str = Field(description="name of the character")
    description: str = Field(description="description of the character")

class Characters(BaseModel):
    characters: List[Character] = Field(description="list of characters and their descriptions")

class NarrativeThread(BaseModel):
    thread: str = Field(description="a list of NARRATIVE THREADS expressed as TROPES that use the CHARACTERS inside the CONTEXT that serve the overall TELEOLOGY of the play. They must be in this format: ACTION between ACTORS in CONTEXT help attain TELEOLOGY in the following way: REASON.")

class NarrativeThreads(BaseModel):
    threads: List[NarrativeThread] = Field(description="List of narrative threads.  They must be in this format: ACTION between ACTORS in CONTEXT help attain TELEOLOGY in the following way: REASON. Must be at least 10 of them")

class TCCN(BaseModel):
    teleology: str = Field(description="the ultimate finality of the play, what would correspond to fate inside the story, or to the moral to a fable, or the ethical teaching to a parable. ")
    context: str = Field(description="a description of the long scale dealings surrounding the story. This may be thought as worldbuilding, and is the overall background where the play develops")
    characters: Characters = Field(description="actors that opulate the world of the play. These must be diverse and must express different sides of the TELEOLOGY. They must be generated as to enable the maximum contrast and dynamical potential in their pairwise interactions when these are made to interact in SCENES.")
    narrative_threads: NarrativeThreads = Field(description="a list of NARRATIVE THREADS expressed as TROPES that use the CHARACTERS inside the CONTEXT that serve the overall TELEOLOGY of the play. You must generate at least 10 threads.")

class Seeding:
    def __init__(self, strong_llm):
        self.strong_llm = strong_llm
        with open(file="prompts/generators/INITIAL_HISTORY_TCC_GENERATOR.txt", mode="r+") as f:
            self.initial_history_tcc_generator = f.read()

        self.tccn_parser = PydanticOutputParser(pydantic_object=TCCN)
        self.seed_template = PromptTemplate(
            template=self.initial_history_tcc_generator,
            input_variables=["seed_description"],
            partial_variables={"format_instructions": self.tccn_parser.get_format_instructions()},
        )
        self.seed_chain = LLMChain(llm=self.strong_llm, prompt=self.seed_template)
        self.db = Database()

    def persist_seed(self, seed_response):
        self.db.add_tccn(seed_response)

    def generate_seed(self, seed_description):
        seed_response = self.seed_chain.run(seed_description)
        return seed_response
    
    def parse_seed(self, seed_description):
        seed_dictionary = self.tccn_parser.parse(seed_description).dict()
        return seed_dictionary
    
    def persist_seed(self, seed_dictonary):
        self.db.add_tccn(seed_dictonary)
    

# %%
strong_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# %%
seed_description = """
Contemporary Lima, Peru. A young man with grandeur fever thinks he will revolutionize philosophy \
by discovering a third logical value and the precise structure of logic when there is no excluded third. \
Satirical, in a John Kennedy Toole way, a bit picaresque, shows the many facets of Lima, its dangers, \
its dearies, its dreams. It all ends up when, in a supremum of irony, the young man dies rolled by a combi \
just before comprehending the ultimate nature of life.
"""

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
seed = Seeding(strong_llm=strong_llm)
tcc_context = seed.generate_seed(seed_description)

# %%
tcc_dictionary = seed.parse_seed(tcc_context)

# %%
for element in tcc_dictionary:
    print(element)
    print(tcc_dictionary[element])
    print()

# %%
seed.persist_seed(tcc_dictionary)

# %%



