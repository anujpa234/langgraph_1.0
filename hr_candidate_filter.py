from langchain_openai import ChatOpenAI
from typing import TypedDict
from langgraph.graph import StateGraph,END,START
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

model= ChatOpenAI(model="gpt-3.5-turbo")

class AgentState(TypedDict):
    application:str
    experience_level:str
    skill_match:str
    response:str

class TopicSelector(BaseModel):
    category: str = Field(description= "Catagorized the candidate based on his profile information")
    reason: str = Field(description="justification, why the candidate is added under cetain category")

parser = PydanticOutputParser(pydantic_object=TopicSelector)

workflow = StateGraph(AgentState)  

def categorize_experience(state: AgentState):
    print("\n Categorize candidate")  
    application=state["application"]

    template = """
                You are an expert career analyst.

                Based on the following job application, categorize the candidate as 'Entry-Level', 'Mid-Level' or 'Senior-Level'.

                Return your response as a JSON object in the following format:
                {format_instructions}

                Application: {application}
                """
    
    prompt = PromptTemplate(
        template=template,
        input_variable=["application"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    ) 

    chain = prompt | model | parser
    experience_level = chain.invoke({"application": application})
    
    return {"experience_level":experience_level.category}

def assess_skillset(state: AgentState):
    print("\n Assessing candidate skillset") 
    question=state["application"]

    template="""
    You are an expert career analyst.
    Based on the following job application of a Python developer, assess the candidate's skillset as 'Matched' or 'Non-Matched'. 
    Return your response as a JSON object in the following format:
    {format_instructions}
    "Application: {application}"    
"""

    prompt = PromptTemplate(
        template=template,
        input_variable=["application"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    skill_match = chain.invoke({"application": question})
    
    return {"skill_match":skill_match.category}

def schedule_hr_interview(state: AgentState):
    print("\n Scheduling Interview")
    return {"response":"Candidate has been shortlisted for HR interview"}

def escalate_to_recruiter(state: AgentState):
    print("\n Escalating to recruiter")
    return {"response":"Candidate has senior-level experience but doesn't match job skills"}

def reject_application(state: AgentState):
    print("\n Sending rejection email")
    return {"response":"Candidate doesn't meet JD and has been rejected"}

def router(state: AgentState):
    skill_match = state["skill_match"].strip().lower()
    experience_level = state["experience_level"].strip().lower()

    if skill_match == "match":
        return "schedule_hr_interview"
    elif experience_level == "senior-level":
        return "escalate_to_recruiter"
    else:
        return "reject_application"

workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skillset)
workflow.add_node("schedule_hr_interview", schedule_hr_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)


workflow.add_edge("categorize_experience", "assess_skillset")

path_map = {
    "schedule_hr_interview": "schedule_hr_interview",
    "escalate_to_recruiter": "escalate_to_recruiter",
    "reject_application": "reject_application"
}
workflow.add_conditional_edges("assess_skillset", router, path_map)

workflow.add_edge(START, "categorize_experience")
workflow.add_edge("assess_skillset",END)
workflow.add_edge("schedule_hr_interview",END)
workflow.add_edge("escalate_to_recruiter",END)
workflow.add_edge("reject_application",END) 
app=workflow.compile()

def run_candidate_screening(application:str):
    results=app.invoke({"application":application})
    return {
        "experience_level":results["experience_level"],
        "skill_match":results["skill_match"],
        "response":results["response"]
    }


application_text = input("Enter the candidate profile information: \n")
results = run_candidate_screening(application_text)
print("\n\nComputed Results :")
print(f"Application: {application_text}")
print(f"Experience Level: {results['experience_level']}")
print(f"Skill Match: {results['skill_match']}")
print(f"Response: {results['response']}")

