from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from litellm import completion
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

class GroqChatModel(BaseChatModel):
    """Custom Groq chat model implementation for CrewAI"""
    
    def __init__(self, api_key: str, model: str = "groq/mixtral-8x7b-32768", temperature: float = 0.7):
        super().__init__()
        os.environ['GROQ_API_KEY'] = api_key

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "groq"
        
    def _generate(self, messages, stop=None, **kwargs):
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
        
        response = completion(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    async def _agenerate(self, messages, stop=None, **kwargs):
        raise NotImplementedError("Async generation not implemented")

@CrewBase
class LatestAiDevelopmentCrew:
    """LatestAiDevelopment crew"""
    
    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"
    
    def __init__(self):
        # Initialize custom Groq chat model
        self.llm = LLM(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="groq/mixtral-8x7b-32768",
            temperature=0.7
        )
        self.final_output = None
        self.docs_scrape_tool = ScrapeWebsiteTool(
            website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
        )
    
    @before_kickoff
    def prepare_inputs(self, inputs):
        inputs['additional_data'] = "Some extra information"
        return inputs
    
    @after_kickoff
    def process_output(self, output):
        output.raw += "\nProcessed after kickoff."
        self.final_output = output.raw
        print("\n=== Final Output ===\n")
        print(self.final_output)
        return output
    
    @agent
    def supporter(self) -> Agent:
        return Agent(
            config=self.agents_config['supporter'],
            verbose=True,
            tools=[],
            llm=self.llm
        )
    
    @agent
    def quality_support_assurer(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_support_assurer'],
            verbose=True,
            llm=self.llm
        )
    
    @task
    def inquiry_resolution_task(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_resolution_task'],
            tools=[self.docs_scrape_tool]
        )
    
    @task
    def quality_assurance_review(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_review']
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.supporter(),
                self.quality_support_assurer()
            ],
            tasks=[
                self.inquiry_resolution_task(),
                self.quality_assurance_review()
            ],
        )

def main():
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    inputs = {
        "customer": "DeepLearningAI",
        "person": "Andrew Ng",
        "inquiry": "I need help with setting up a Crew "
                  "and kicking it off, specifically "
                  "how can I add memory to my crew? "
                  "Can you provide guidance in short?"
    }
    
    # Create and run crew instance
    crew_instance = LatestAiDevelopmentCrew()
    result = crew_instance.crew().kickoff(inputs=inputs)
    print(result)
    
    print("\n=== Accessed Final Output ===\n")
    print(crew_instance.final_output)

if __name__ == "__main__":
    main()