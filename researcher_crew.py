from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool
from litellm import completion
import os
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self, api_key, model="groq/mixtral-8x7b-32768", temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        os.environ['GROQ_API_KEY'] = api_key
        
    def call(self, messages):
        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content

@CrewBase
class LatestAiDevelopmentCrew:
    """LatestAiDevelopment crew"""
    
    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"
    
    def __init__(self):
        # Initialize Groq using litellm
        self.llm = LLM(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="groq/mixtral-8x7b-32768",  # Note the 'groq/' prefix
            temperature=0.7
        )
        self.final_output = None
    
    @before_kickoff
    def prepare_inputs(self, inputs):
        # Modify inputs before the crew starts
        inputs['additional_data'] = "Some extra information"
        return inputs
    
    @after_kickoff
    def process_output(self, output):
        # Modify output after the crew finishes
        output.raw += "\nProcessed after kickoff."
        self.final_output = output.raw
        print("\n=== Final Output ===\n")
        print(self.final_output)
        return output
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()],
            llm=self.llm
        )
    
    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            llm=self.llm
        )
    
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )
    
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task']
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.researcher(),
                self.reporting_analyst()
            ],
            tasks=[
                self.research_task(),
                self.reporting_task()
            ],
            process=Process.sequential
        )

def main():
    # Test the LLM setup first
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    # Test basic completion
    try:
        test_response = completion(
            model="groq/mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "Hello from litellm"}]
        )
        print("Test response:", test_response)
    except Exception as e:
        print("Test failed:", str(e))
        return
    
    inputs = {
        'topic': 'Give brief information about knowledge graphs in 100 words'
    }
    
    # Create crew instance
    crew_instance = LatestAiDevelopmentCrew()
    
    # Run the crew
    result = crew_instance.crew().kickoff(inputs=inputs)
    print(result)
    
    # Access the final output if needed
    print("\n=== Accessed Final Output ===\n")
    print(crew_instance.final_output)

if __name__ == "__main__":
    main()