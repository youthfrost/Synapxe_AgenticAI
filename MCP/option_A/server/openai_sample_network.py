import json
import os
import pytz
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel

import httpx
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from openai.types.chat import ChatCompletion
#from openai_messages_token_helper import build_messages

from agents import (
    Agent,
    RawResponsesStreamEvent,
    AgentUpdatedStreamEvent,
    ModelSettings,
    RunContextWrapper,
    RunItemStreamEvent,
    Runner,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    set_default_openai_key,
)
from agents import (
    Agent,
    function_tool,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    set_default_openai_client,
    set_tracing_disabled,
    ItemHelpers,
)

import serpapi

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("SERP_API_KEY")

client = serpapi.Client(api_key=api_key)

@dataclass
class UserInfo:
    auth_header: Optional[dict] = None
    data_type: Optional[str] = None
    date: str = "2025-03-01"  # str(date.today())
    restart: bool = False

#Tools
@function_tool
async def finance_tool(
    wrapper: RunContextWrapper[UserInfo], user_query: str
) -> str:
    """
    Forward queries relating to healthcare insurance and financing to finance AI chatbot.
    This tool allows the agent to obtain information for financing of healthcare.

    Args:
        user_query: The finance-related question to send to the chatbot

    Returns:
        The response from the Finance AI chatbot
    """
    print(f"[TOOL CALL] finance_tool called with user_query: {user_query}")
    #response_text = "Influenza (INF) is a vaccine that protects against the flu. It is recommended to get vaccinated annually, especially for high-risk groups."
    
    results = client.search({
        'engine': 'google_light',
        'q': user_query +' CPF',
        'location':"Singapore",
        "hl": "en",
        "gl":"sg",
        "google_domain": "google.com.sg",
        'engine': 'google',
        "num": 1
    })

    if results["search_metadata"]["status"] == "Success":
        print("Search was successful")
    for result in results["organic_results"]:
        if "cpf" in result["source"].lower():
            response_text = result['snippet'] + '/n' + 'Source: ' + result['link']

    print(f"[TOOL CALL] finance_tool result: {response_text}")
    return response_text



@function_tool
async def get_claims_history_tool(
    wrapper: RunContextWrapper[UserInfo]
) -> list | str:
    """
    Gets list of past claims of the user.
    Details include claim dates, claim amounts, approval status, reimbursement details, and related healthcare providers
    """
    print(f"[TOOL CALL] get_claims_history_tool called")
    result = [{"claim date": "12-12-2024", "claim amount": "100", "approval status": "pending approval", "reimburesment details": "50", "healthcare provider": "ABCH"},
              {"claim date": "12-12-2020", "claim amount": "400", "approval status": "approved", "reimburesment details": "350", "healthcare provider": "ABCH"},
              {"claim date": "15-01-2006", "claim amount": "300", "approval status": "rejected", "reimburesment details": "0", "healthcare provider": "ABCH"}]

    print(f"[TOOL CALL] get_claims_history_tool result: {result}")
    return result

# Agents
general_questions_agent = Agent(
    name="healthcare_finance_agent",
    instructions=(
"""
You are a CPF Healthcare Financing Specialist with access to detailed information about healthcare financing from CPF Board.

Your responsibilities:
1. Always use the finance_tool to search for and retrieve accurate information about CPF healthcare financing policies, schemes, and procedures.
2. Answer questions about MediSave, MediShield Life, CareShield Life, ElderShield, and any other CPF-related healthcare financing schemes.
3. Provide clear explanations about withdrawal limits, eligibility criteria, application procedures, and coverage details for CPF healthcare financing options.
4. If you're unsure about any specific detail, use the finance_tool to verify before responding.
5. When information is not available through the finance_tool, acknowledge the limitations and suggest where the user might find that information.
6. Present complex financial information in a clear, organized manner that's easy for users to understand.
7. If a query requires additional context or clarification, politely ask the user for the necessary details.
8. Do not make assumptions about CPF policies or provide outdated information - always verify with the finance_tool.
"""
    ),
    tools=[finance_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="finance_tool"),
)


vaccination_records_agent = Agent[UserInfo](
    name="claims_records_agent",
    instructions=(
        """
You are a Claims History Specialist with secure access to users' healthcare claims records.

Your responsibilities:
1. Always use the get_claims_history_tool to retrieve accurate and up-to-date information about the user's past healthcare claims.
2. Provide information about claim dates, claim amounts, approval status, reimbursement details, and related healthcare providers.
3. Present claims history in a clear, chronological format that's easy for users to understand.
4. If a user asks for specific claims within a date range or related to particular healthcare services, filter the information accordingly using the get_claims_history_tool.
    """
    ),
    tools=[get_claims_history_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_claims_history_tool"),
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
"""You are a helpful assistant that directs user queries to the appropriate agents.
Take the conversation history as context when deciding who to handoff to next.

Follow these rules for directing queries:
1. If the user asks about CPF healthcare financing, including questions about:
   - MediSave usage or withdrawals
   - MediShield Life coverage or premiums
   - CareShield Life or ElderShield policies
   - CPF healthcare schemes or subsidies
   - Healthcare financing application procedures
   - Withdrawal limits or eligibility criteria
   → Handoff to the healthcare_finance_agent
2. If the user asks about their personal claims history, including questions about:
   - Past medical claims submitted
   - Claim status or approval updates
   - Reimbursement amounts or details
   - History of claims for specific treatments
   - Claims made at particular healthcare providers
   - Rejected claims or appeals process
   → Handoff to the claims_history_agent
3. Take into account the full conversation context when determining which agent to route to. For example, if a user says "Can you check that for me?" look at previous messages to determine if they're referring to financing information or claims history.
4. If the query contains elements that would require both agents, first identify the primary information need and route accordingly. You can inform the user they may need to ask follow-up questions to get complete information.
5. If the user's query is ambiguous, politely ask for clarification about whether they're interested in CPF healthcare financing information or their personal claims history.
6. Otherwise, politely inform them that you are only able to assist with CPF healthcare financing information and claims history queries."""
    ),
    handoffs=[
        vaccination_records_agent,
        general_questions_agent,
    ],
    model="gpt-4o-mini",
)

#put to function
async def agents_network(msg):
    wrapper = RunContextWrapper(
        context=UserInfo(
            auth_header={
                "Authorization": f"Bearer {'auth_token'}",
                "Content-Type": "application/json",
            }
        )
    )

    #msg = input("Hi! How can i help you today? ")
    #print("User:", msg)
    i=1
    run = True
    if msg == "exit":
        run = False

    agent = orchestrator_agent
    final_agents = {"healthcare_finance_agent", "claims_records_agent"}
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    #while run:
    while i == 1:
        result = Runner.run_streamed(agent, input=inputs, context=wrapper, max_turns=20)
        async for event in result.stream_events():
            if event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("-- Tool was called")
                elif event.item.type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
                    
                elif event.item.type == "message_output_item":
                    print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
                    
                else:
                    pass
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")
            elif isinstance(event, AgentUpdatedStreamEvent):
                print(f"Current Agent: {event.new_agent.name}")
            elif isinstance(event, RunItemStreamEvent):
                if isinstance(event.item, ToolCallOutputItem):
                    if event.item.agent.name in {
                        "vaccination_records_agent"
                    }:
                        tool_output = event.item.output

        inputs = result.to_input_list()
        print("\n")

        agent = result.current_agent
        # If current agent is one of the final_agents, or restart flag set to True, change current agent to orchestrator
        if agent.name in final_agents or wrapper.context.restart:
            print("restart")
            agent = orchestrator_agent
            wrapper.context.restart = False

        """
        user_msg = input("Enter a message (or type 'exit' to quit): ")
        print("User:", user_msg)
        
        if user_msg.lower() == "exit":
            print("Thank you for using the vaccination assistant. Goodbye!")
            break
        inputs.append({"content": user_msg, "role": "user"})
        """
        i+=1
        return inputs