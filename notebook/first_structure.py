# %%
from pydantic import BaseModel, Field
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()


class Character(BaseModel):
    name: str = Field(description="name of the character")
    description: str = Field(description="description of the character")


class Characters(BaseModel):
    characters: List[Character] = Field(description="list of characters and their descriptions")


class Seeding:
    def __init__(self, strong_llm):
        self.strong_llm = strong_llm
        with open(file="prompts/generators/INITIAL_HISTORY_TCC_GENERATOR.txt", mode="r+") as f:
            self.initial_history_tcc_generator = f.read()
        self.seed_template = PromptTemplate(
            template=self.initial_history_tcc_generator,
            input_variables=["seed_description"]
        )
        self.seed_chain = LLMChain(llm=self.strong_llm, prompt=self.seed_template)

    def generate_seed(self, seed_description):
        seed_response = self.seed_chain.run(seed_description)
        return seed_response
    

class CharacterParser:
    def __init__(self, fast_llm):
        self.fast_llm = fast_llm
        self.character_parser = PydanticOutputParser(pydantic_object=Characters)
        self.template = PromptTemplate(
            template="Parse the character information from the context.\n{format_instructions}\n{context}\n",
            input_variables=["context"],
            partial_variables={"format_instructions": self.character_parser.get_format_instructions()},
        )
        self.character_parser_chain = LLMChain(llm=self.fast_llm, prompt=self.template)

    def parse_characters(self, context):
        parsed_output = self.character_parser_chain.run(context=context)
        characters = self.character_parser.parse(parsed_output).dict()
        return characters
    
    
class ForwardPass:
    def __init__(self, fast_llm):
        self.fast_llm = fast_llm
        self.character_parser = CharacterParser(self.fast_llm)
    
    def parse_characters(self, seed_response):
        character_dictionary = self.character_parser.parse_characters(seed_response)
        return character_dictionary


class BackwardPass:
    def __init__(self, strong_llm):
        self.strong_llm = strong_llm
        with open(file="prompts/generators/FIRST_PASS_CHARACTER_DESIGNER.txt", mode="r+") as f:
            self.first_pass_character_designer = f.read()
        self.character_template = PromptTemplate(
            template=self.first_pass_character_designer,
            input_variables=["tcc_context", "character_description"]
        )
        self.character_chain = LLMChain(llm=self.strong_llm, prompt=self.character_template)

    def design_characters(self, seed_description, character_dictionary):
        character_info = {}
        for character in character_dictionary['characters']:
            char_name = character['name']
            print("Processing character", char_name)
            character_description = str(character)
            character_response = self.character_chain.run(
                dict(tcc_context=seed_description, 
                    character_description=character_description)
            )
            character_info[char_name] = {
                'name': char_name,
                'epoch': 1,
                'HPPTI': character_response
            }
        return character_info


class SceneGenerator:
    def __init__(self, llm):
        with open(file="prompts/generators/INITIAL_SCENE_TCC_GENERATOR.txt", mode="r+") as f:
            self.initial_scene_tcc_generator = f.read()
        self.llm = llm
        self.template = PromptTemplate(
            template=self.initial_scene_tcc_generator,
            input_variables=["tcc_context"]
        )

    def generate_scenes(self, seed_response):
        return self.llm.run(self.template, seed_response)


class SceneEvaluator:
    def __init__(self, llm):
        with open(file="prompts/generators/INITIAL_SCENE_EVALUATOR_TCC_GENERATOR.txt", mode="r+") as f:
            self.initial_scene_evaluator_tcc_generator = f.read()
        self.llm = llm
        self.template = PromptTemplate(
            template=self.initial_scene_evaluator_tcc_generator,
            input_variables=["scene_context"]
        )

    def evaluate_scenes(self, scene_response):
        return self.llm.run(self.template, scene_response)


class FirstPassWriter:
    def __init__(self, llm):
        with open(file="prompts/generators/FIRST_PASS_WRITER_TCC_TEMPLATE.txt", mode="r+") as f:
            self.first_pass_writer_tcc_template = f.read()
        self.llm = llm
        self.template = PromptTemplate(
            template=self.first_pass_writer_tcc_template,
            input_variables=["scene_context"]
        )

    def write_first_draft(self, scene_response):
        return self.llm.run(self.template, scene_response)


class PlayWriter:
    def __init__(self):
        self.strong_llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        self.fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        self.seeding = Seeding(self.strong_llm)
        self.forward_pass = ForwardPass(self.fast_llm)
        self.backward_pass = BackwardPass(self.strong_llm)
        self.scene_generator = SceneGenerator(self.strong_llm)
        self.scene_evaluator = SceneEvaluator(self.strong_llm)
        self.first_pass_writer = FirstPassWriter(self.strong_llm)

    def execute(self, seed_description):
        seed_response = self.seeding.generate_seed(seed_description)
        print(seed_response)

        character_dictionary = self.forward_pass.parse_characters(seed_response)
        print(character_dictionary)

        character_info = self.backward_pass.design_characters(seed_description, character_dictionary)
        print(character_info)

        scene_response = self.scene_generator.generate_scenes(character_info)
        print(scene_response)

        evaluator_response = self.scene_evaluator.evaluate_scenes(scene_response)
        print(evaluator_response)

        writer_response = self.first_pass_writer.write_first_draft(evaluator_response)
        print(writer_response)

        playwrite = {
            'seed_response': seed_response,
            'character_dictionary': character_dictionary,
            'character_info': character_info,
            'scene_response': scene_response,
            'evaluator_response': evaluator_response,
            'writer_response': writer_response
        }
        return playwrite 

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
playwriter = PlayWriter()
play = playwriter.execute(seed_description)


# %%
character_dictionary = {'Protagonist': {'name': 'Protagonist', 'epoch': 1, 'HPPTI': "1. History: The protagonist was born in an abandoned skyscraper in a magical cyberpunk city, a place where the old and new world collide. His parents were unknown, and he was raised by the city's outcasts and misfits, learning from their wisdom and experiences. His life took a dramatic turn when he discovered his ability to control stone giants, a power once possessed by the city's old dead emperor. This revelation marked him as the emperor's successor, and he was thrust into a position of power and responsibility. Despite his youth, he was forced to grapple with complex issues of power, mortality, and the fate of his city.\n\n2. Physical Condition: The protagonist is a frail boy, appearing younger than his actual age due to his slight build and delicate features. He has dark hair and bright, intelligent eyes that seem to hold a wisdom beyond his years. Despite his physical frailty, he possesses an inner strength that is evident in his determined gaze and the way he carries himself. His health is generally good, but his frailty makes him vulnerable to physical harm. His most distinguishing feature is his ability to command stone giants, a power that manifests as a glowing aura around him when he uses it.\n\n3. Philosophy: The protagonist believes in the inherent value of all life and the importance of using power responsibly. He is deeply aware of the impermanence of life and the inevitability of death, and this awareness shapes his worldview and actions. He values wisdom, compassion, and justice, and strives to uphold these values in his decisions and actions. He believes that power should be used for the benefit of all, not for personal gain or domination. He also believes that every individual has a role to play in the world, and that his role is to protect his city and its inhabitants.\n\n4. Teleology: The protagonist's primary goal is to protect his city and its inhabitants from harm. He is motivated by a deep sense of responsibility and love for his city, and he is willing to make great sacrifices to ensure its survival. His long-term goal is to find a way to use his powers to create a better future for his city, one where power is used responsibly and for the benefit of all. He is also driven by a desire to understand the nature of his powers and the legacy of the old emperor, hoping to learn from the past to shape the future.\n\n5. Internal Contradiction: The protagonist's internal contradiction lies in the tension between his immense power and his frailty. Despite his ability to control stone giants, he is physically vulnerable and aware of his mortality. This contradiction creates a constant struggle within him, as he grapples with the fear of death and the burden of his powers. He also struggles with the moral dilemma of using his powers, as he is aware of the potential for destruction and harm. This internal conflict adds depth to his character and creates potential for growth and development."}, 'Old Emperor': {'name': 'Old Emperor', 'epoch': 1, 'HPPTI': "1. History: The Old Emperor, whose real name has been lost to time, was once a powerful ruler of the magical cyberpunk city. Born into a lineage of stone giant controllers, he ascended to the throne at a young age and ruled with a firm but fair hand. His reign was marked by prosperity and peace, as he used his powers to protect the city and its inhabitants. However, his rule was also marked by isolation and loneliness, as his powers set him apart from others and made him a target for those who coveted his abilities. He spent his final years in the old temple in the middle of the city, where he eventually died alone. His spirit now appears in dreams and visions, guiding the protagonist who has inherited his powers.\n\n2. Physical Condition: As a spirit, the Old Emperor does not have a physical form in the traditional sense. However, in dreams and visions, he appears as he did in life: an elderly man of average height and build, with a long white beard and deep-set eyes that are a striking shade of blue. His most distinguishing feature is the crown he wears, a symbol of his former status as ruler of the city. Despite his age, he exudes an aura of strength and vitality, a testament to his former powers. \n\n3. Philosophy: The Old Emperor believed in the importance of using power responsibly and for the greater good. He valued peace and prosperity, and saw his powers as a tool to protect and serve his city. However, he also understood the dangers of power, and the isolation and loneliness that can come with it. He believed in the inevitability of death, and saw it as a release from the burdens of power and the struggles of life. \n\n4. Teleology: The Old Emperor's primary goal is to guide the protagonist in using his inherited powers. He wants to ensure that the protagonist uses his powers responsibly and for the greater good, just as he did. He also wants to help the protagonist navigate the challenges and dangers that come with power, and to prepare him for the difficult choices he will have to make. \n\n5. Internal Contradiction: The Old Emperor's belief in the importance of using power for the greater good is in conflict with his desire to protect the protagonist from the burdens and dangers of power. He wants the protagonist to use his powers responsibly, but he also wants to shield him from the isolation and loneliness that he himself experienced. This contradiction creates a tension within him, as he struggles to balance his duty as a guide with his desire to protect."}, 'Stone Giants': {'name': 'Stone Giants', 'epoch': 1, 'HPPTI': "1. History: The Stone Giants have a long and mysterious history that predates even the oldest records of the city. They were created by the old dead emperor, a powerful figure who used his magic to shape the giants from the city's bedrock. The emperor used the giants to build and protect the city, and they became symbols of his power and authority. When the emperor died, the giants fell into a deep slumber, waiting for a new master to awaken them. They were discovered by the protagonist, a young boy who inherited the emperor's powers, in the abandoned skyscraper where he was born. The boy's connection with the giants was immediate and profound, and he quickly learned to command them.\n\n2. Physical Condition: The Stone Giants are massive creatures made entirely of stone. They stand several stories tall, with broad shoulders, powerful limbs, and faces that are both stern and serene. Their bodies are rough and weathered, covered in cracks and moss that hint at their great age. Despite their size and weight, the giants move with surprising grace and agility. They are incredibly strong and durable, capable of withstanding attacks that would destroy lesser beings. However, they are not invincible, and can be damaged or destroyed by powerful magic or weapons.\n\n3. Philosophy: As creatures of stone, the giants do not have thoughts or emotions in the same way that humans do. However, they possess a deep sense of duty and loyalty to their master. They follow the protagonist's commands without question, and will protect him and the city at all costs. They do not seek power or glory for themselves, but exist solely to serve. Despite their silence, the giants are capable of expressing a range of emotions through their actions and body language, from determination and resolve to sorrow and regret.\n\n4. Teleology: The Stone Giants' primary goal is to obey and protect their master. They are motivated by a deep-seated instinct to serve, and will do whatever is necessary to fulfill their duties. They are willing to fight, toil, and even sacrifice themselves for the sake of their master and the city. Their long-term objective is to ensure the safety and prosperity of the city, which they view as their home and responsibility.\n\n5. Internal Contradiction: The Stone Giants' greatest contradiction lies in their relationship with the protagonist. On one hand, they are bound to obey his commands and protect him at all costs. On the other hand, they are aware of the destructive potential of their own power, and the danger it poses to the city and the world. This creates a tension between their duty to their master and their desire to prevent harm. As the protagonist grows older and his decisions become more complex and morally ambiguous, this tension will only increase, leading to potential conflict and growth."}, 'Tech Hacker': {'name': 'Tech Hacker', 'epoch': 1, 'HPPTI': '1. History: Tech Hacker, originally named Orion, was born and raised in the heart of the cyberpunk city. His parents were both engineers, and from a young age, Orion showed a natural aptitude for technology. He was a prodigy, hacking into the city\'s mainframe when he was just ten. However, his parents were killed in a mysterious accident when he was a teenager, which led him to question the city\'s ruling powers. He became a rebel, using his tech skills to expose corruption and fight against the city\'s oppressive regime. Despite his rebellious nature, Orion was always a loner, preferring the company of machines to people. That was until he crossed paths with the protagonist, a young boy with unimaginable powers.\n\n2. Physical Condition: Orion, now known as Tech Hacker, is in his late twenties. He stands at 5\'10" and has a lean, wiry build, a testament to his life on the run. His hair is jet black, often hidden under a hood or a beanie, and his eyes are a piercing blue. He has a few noticeable scars, the most prominent one running down his left cheek, a souvenir from a run-in with the city\'s enforcers. Despite his rough exterior, Orion is in good health and has exceptional agility and reflexes, honed from years of evading capture.\n\n3. Philosophy: Tech Hacker believes in freedom and justice. He values truth and transparency, and despises those who abuse power. He is a firm believer in the power of technology as a tool for change and sees himself as a guardian of the city\'s oppressed inhabitants. Despite his initial skepticism towards the protagonist, he eventually comes to see the potential for good in the boy\'s powers.\n\n4. Teleology: Tech Hacker\'s primary goal is to expose the city\'s corruption and bring about a revolution. He is driven by a desire for justice and a deep-seated anger towards those who abuse power. His secondary goal is to protect the protagonist, whom he sees as a key to the city\'s salvation. Despite his rebellious nature, Orion is willing to work with others and make sacrifices for the greater good.\n\n5. Internal Contradiction: Tech Hacker\'s internal contradiction lies in his relationship with technology. While he sees it as a tool for change and liberation, he also recognizes its potential for destruction and oppression. This conflict is further complicated by his alliance with the protagonist, whose powers are both awe-inspiring and terrifying. Despite his belief in the boy\'s potential for good, Orion can\'t help but fear the destruction he could cause. This contradiction creates a tension within him, as he struggles to reconcile his belief in the power of technology with his fear of its potential for destruction.'}, 'Mystic Seer': {'name': 'Mystic Seer', 'epoch': 1, 'HPPTI': "1. History: Mystic Seer was born in the outskirts of the city, in a humble family. Her parents were simple folk who believed in the old ways and the power of the ancient emperor. From a young age, she showed signs of possessing a deep understanding of the world and the forces that govern it. She was taken under the wing of the city's oldest seer, who recognized her potential and trained her in the ways of divination and guidance. Over the years, she became a respected figure in the city, known for her wisdom and her ability to see beyond the surface of things. She has guided many individuals on their paths, but none as significant as the young boy who would grow to command the stone giants.\n\n2. Physical Condition: Mystic Seer is an elderly woman, around 80 years old. She is of average height and has a slender build, a testament to her simple lifestyle and diet. Her hair is long and white, often tied in a loose bun, and her eyes are a deep, penetrating blue. Her skin is wrinkled from age, but her posture is straight, and she moves with a grace that belies her years. Despite her age, she is in good health, although she does have a slight limp from an old injury. Her most distinguishing feature is a tattoo of an eye on her forehead, a symbol of her role as a seer.\n\n3. Philosophy: Mystic Seer believes in the interconnectedness of all things and the cyclical nature of life and death. She values wisdom, humility, and balance above all else. She believes that power is a tool, not an end in itself, and that it should be used wisely and responsibly. She sees the world as a complex web of relationships and forces, and believes that understanding these connections is the key to navigating life's challenges. She believes in the inherent worth of every individual and the potential for growth and change.\n\n4. Teleology: Mystic Seer's primary goal is to guide the protagonist on his journey, helping him understand his powers and the choices he must make. She is motivated by a deep sense of duty and a belief in the protagonist's potential to bring balance to the city. She is willing to sacrifice her own well-being to ensure the protagonist's success, and she is committed to helping him navigate the difficult path ahead of him.\n\n5. Internal Contradiction: Despite her belief in the interconnectedness of all things and the potential for growth and change, Mystic Seer struggles with the inevitability of her own death. She knows that she will not live to see the outcome of the protagonist's journey, and this knowledge creates a tension between her duty as a guide and her desire to witness the fruition of her efforts. This contradiction adds depth to her character and creates potential for internal conflict and growth."}, 'City Mayor': {'name': 'City Mayor', 'epoch': 1, 'HPPTI': "1. History: The City Mayor, whose name is Arlen, was born into a family of politicians. His father was a senator, and his mother was a high-ranking city official. Arlen was groomed from a young age to take on a leadership role, and he was educated at the city's most prestigious schools. He was elected mayor at a relatively young age, and he has held the position for several terms. Arlen has always been a pragmatic and ambitious leader, but his fear of the protagonist's power has led him to become increasingly paranoid and controlling. He has seen the devastation caused by the old emperor's stone giants, and he is determined to prevent a similar catastrophe.\n\n2. Physical Condition: Arlen is in his late fifties, but he looks older due to the stress of his position. He stands at an average height and has a lean build. His hair is silver and neatly combed back, and his eyes are a piercing blue. He has a sharp, angular face with a prominent nose and thin lips. Despite his age, Arlen is in good health and maintains a strict fitness regimen. He has no physical disabilities, but he does suffer from insomnia and occasional bouts of anxiety.\n\n3. Philosophy: Arlen believes in the importance of order and stability. He values the rule of law and is deeply committed to the welfare of his city. He views the protagonist's powers as a threat to this order, and he believes that it is his duty to control or neutralize this threat. Arlen is pragmatic and utilitarian in his approach, and he is willing to make difficult decisions for the greater good. He has a deep-seated fear of chaos and destruction, and he is willing to do whatever it takes to prevent this.\n\n4. Teleology: Arlen's primary goal is to maintain the stability and prosperity of his city. He is motivated by a sense of duty and a desire to protect his constituents. He also has a personal goal of controlling the protagonist's powers, as he sees this as the key to ensuring the city's safety. Arlen is willing to use any means necessary to achieve these goals, including manipulation, coercion, and force.\n\n5. Internal Contradiction: Despite his commitment to the rule of law, Arlen is willing to bend or break the rules in order to control the protagonist. This creates a tension between his public persona as a principled leader and his private actions, which are often ruthless and underhanded. This contradiction adds depth to his character and creates potential for internal conflict and growth. It also raises questions about the nature of power and the lengths that people are willing to go to in order to maintain it."}, 'Street Urchin': {'name': 'Street Urchin', 'epoch': 1, 'HPPTI': "1. History: \nStreet Urchin, whose real name is unknown, was born and raised in the underbelly of the magical cyberpunk city. His parents, both petty thieves, died when he was very young, leaving him to fend for himself on the dangerous city streets. He quickly learned to be resourceful and street-smart, using his wits to survive. He met the protagonist when they were both children, and they formed a strong bond. Street Urchin was always fascinated by the old tales of the city's dead emperor and his titanic stone giants. When the protagonist discovered his powers, Street Urchin became his confidant and ally, helping him navigate the complexities of his newfound abilities and the city's political landscape.\n\n2. Physical Condition: \nStreet Urchin is a child of about 12 years old. He is small for his age, with a lean, wiry build that speaks of a life of hardship. He has messy brown hair, bright green eyes, and a face that is perpetually smudged with dirt. Despite his tough upbringing, he is surprisingly healthy and agile, with quick reflexes honed by years of living on the streets. He has a distinctive scar on his left cheek, a souvenir from a close encounter with a city guard.\n\n3. Philosophy: \nStreet Urchin's philosophy is shaped by his harsh life experiences. He believes in survival at all costs and has a pragmatic, no-nonsense approach to life. He values loyalty and friendship above all else, as these are the things that have kept him alive. He is skeptical of authority and power, having seen the corruption and abuse that often accompany them. Despite his tough exterior, he has a deep sense of empathy for others who are struggling, and a strong desire to help those in need.\n\n4. Teleology: \nStreet Urchin's immediate goal is survival. He wants to ensure that he and his friends, especially the protagonist, stay safe in the dangerous city. His long-term goal is to see the city become a better place, free from corruption and oppression. He is motivated by his friendship with the protagonist and his desire to see him use his powers for good. He is willing to risk his own safety to help the protagonist navigate his powers and make the right decisions.\n\n5. Internal Contradiction: \nStreet Urchin's internal contradiction lies in his skepticism of power and his friendship with the protagonist. He distrusts authority and power, yet he supports and aids his friend who has inherited immense power. This creates a tension within him, as he struggles to reconcile his beliefs with his actions. He fears that the protagonist may become corrupted by his powers, yet he also believes in his friend's goodness and potential to bring about positive change. This contradiction adds depth to his character and creates potential for internal conflict and growth."}, 'Cybernetic Enforcer': {'name': 'Cybernetic Enforcer', 'epoch': 1, 'HPPTI': '1. History: The Cybernetic Enforcer, whose real name is lost in the annals of time, was once a street urchin, surviving in the harsh underbelly of the city. His life took a turn when he was picked up by the Mayor\'s men for petty theft. Instead of punishing him, the Mayor saw potential in his ruthless survival instincts and took him under his wing. The Mayor, recognizing the need for a strong hand to maintain his rule, offered the boy a chance to rise above his circumstances. The boy accepted and underwent a series of painful cybernetic enhancements, transforming him into the Mayor\'s right hand and the city\'s most feared enforcer.\n\n2. Physical Condition: The Cybernetic Enforcer is a man in his late 40s, standing at 6\'2" and weighing around 220 pounds. His physical build is muscular and intimidating, a result of rigorous training and cybernetic enhancements. His hair is shaved, and his eyes are a cold steel blue, often hidden behind cybernetic visors. His most distinguishing feature is his right arm, replaced with a powerful cybernetic limb capable of immense strength and precision. Despite his enhancements, he is not invulnerable and has accumulated scars and damage over the years, each a testament to his battles.\n\n3. Philosophy: The Cybernetic Enforcer believes in the survival of the fittest. He views the world as a harsh and unforgiving place where only the strong survive. His loyalty to the Mayor is unwavering, as he sees him as the one who gave him a chance to rise above his circumstances. He believes in maintaining order at all costs, even if it means using fear and intimidation. He views the boy\'s powers as a threat to the city\'s stability and is willing to do whatever it takes to neutralize him.\n\n4. Teleology: His primary goal is to maintain the Mayor\'s rule and the city\'s order. He is motivated by his loyalty to the Mayor and his belief in a structured society. He is willing to use his cybernetic enhancements and ruthless tactics to achieve this goal. His long-term objective is to ensure the city\'s survival, even if it means confronting the boy and his titanic stone giants.\n\n5. Internal Contradiction: Despite his ruthless exterior, the Cybernetic Enforcer struggles with the morality of his actions. He is torn between his loyalty to the Mayor and his growing sympathy for the boy, who reminds him of his own past. This internal conflict adds depth to his character, as he grapples with the question of whether the ends truly justify the means. His public persona as the Mayor\'s ruthless enforcer is at odds with his private self, a man who still remembers the hardships of his past and questions the morality of his actions.'}, 'Ancient Librarian': {'name': 'Ancient Librarian', 'epoch': 1, 'HPPTI': "1. History: The Ancient Librarian, known as Thoth, was once a scholar in the city's grand university. He was fascinated by the city's history and the old emperor's reign. When the emperor died, Thoth dedicated his life to preserving the city's history and understanding the emperor's powers. He moved into the city's library, where he has lived for centuries, sustained by the magic of the city. Thoth has seen the city change and evolve, and he has helped many leaders navigate its challenges. He has always been a solitary figure, preferring the company of books to people. His only friend was the old emperor, and he has felt a deep sense of loss since the emperor's death.\n\n2. Physical Condition: Thoth is an old man, with a frail and thin body. He is tall, with a slight stoop from years of reading and writing. His hair is white and thinning, and his eyes are a pale blue, almost translucent. His skin is wrinkled and pale from spending most of his time indoors. Despite his age and frailty, Thoth is in good health, thanks to the city's magic. He has a sharp mind and keen senses, and he is able to move around the library with ease. His most distinguishing feature is a pair of round glasses that he always wears.\n\n3. Philosophy: Thoth believes in the power of knowledge and the importance of history. He values truth and wisdom above all else, and he is dedicated to preserving the city's history and secrets. He believes that the city's magic is a gift, but also a responsibility. He sees the city as a living entity, with its own soul and consciousness. He believes that the city's wellbeing is tied to the wellbeing of its people, and that the city's leaders have a duty to protect and nurture both.\n\n4. Teleology: Thoth's main goal is to help the protagonist understand his powers and the city's history. He wants to ensure that the city's magic is used wisely and responsibly. He is motivated by his love for the city and his desire to preserve its history. He is willing to risk his own life to help the protagonist, and he is prepared to make difficult decisions for the greater good of the city.\n\n5. Internal Contradiction: Thoth is torn between his duty to the city and his friendship with the old emperor. He feels a deep sense of loyalty to the emperor, but he also believes that the emperor's powers were too great and dangerous. He struggles with the knowledge that the protagonist has inherited these powers, and he fears that the protagonist might misuse them. This internal conflict adds depth to Thoth's character and creates potential for growth and change."}, 'Rebel Leader': {'name': 'Rebel Leader', 'epoch': 1, 'HPPTI': "1. History: The Rebel Leader, whose real name is Arlen, was born and raised in the underbelly of the city, where the poor and downtrodden reside. His parents were laborers, working tirelessly to provide for their family. However, they were killed in a factory accident when Arlen was just a teenager, leaving him to fend for himself. This event sparked a deep resentment in him towards the city's leadership, who he believed were exploiting the poor for their own gain. He educated himself, reading books and learning about the city's history and politics. As he grew older, he formed a resistance group with the aim of overthrowing the city's leadership and establishing a more equitable society.\n\n2. Physical Condition: Arlen is in his mid-40s, standing at a height of 6 feet with a muscular build from years of physical training and combat. He has a rugged appearance, with a scar running down his left cheek, a reminder of a battle fought years ago. His hair is dark and cropped short, and his eyes are a piercing blue. Despite his tough exterior, he has a warm smile that can put anyone at ease. He is in excellent physical condition, with high endurance and agility, but years of fighting have left him with a slight limp in his right leg.\n\n3. Philosophy: Arlen believes in equality and justice for all. He is deeply critical of the city's leadership, viewing them as corrupt and self-serving. He believes that power should be used for the benefit of all, not just a select few. He values honesty, loyalty, and courage, and is willing to put his life on the line for his beliefs. He is skeptical of the protagonist's powers, fearing that they could be used to further oppress the people. However, he also sees the potential for these powers to bring about change.\n\n4. Teleology: Arlen's primary goal is to overthrow the city's leadership and establish a more equitable society. He is motivated by his desire for justice and his belief in a better future for the city's residents. He is willing to do whatever it takes to achieve this goal, even if it means forming an alliance with the protagonist. His long-term goal is to create a society where everyone has access to basic necessities and opportunities for growth and development.\n\n5. Internal Contradiction: Arlen's internal contradiction lies in his desire for change and his fear of the protagonist's powers. While he recognizes the potential for these powers to bring about the change he desires, he is also wary of the potential for abuse. This creates a tension within him, as he struggles to decide whether to ally with the protagonist or view him as a threat. This contradiction adds depth to his character, as it forces him to grapple with his beliefs and fears, and could potentially lead to growth and change."}}

# %%
for character in character_dictionary.keys():
    print("========================")
    print(character_dictionary[character]['name'])
    print(character_dictionary[character]['HPPTI'])
    print("")
    print("========================")

# %%



