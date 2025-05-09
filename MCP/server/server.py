#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastMCP server application exposing AI agent–powered services for:
  1. Appointment Booking (BookingService)
  2. General Information Enquiries (GeneralInfoService), e.g. vaccine side effects

Usage:
    python server.py

Dependencies:
    - fastmcp
    - openai-agent-sdk
"""
import asyncio
import json
import os
import pytz
import random
import urllib.parse
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai_messages_token_helper import build_messages

from agents import (
    Agent,
    AgentUpdatedStreamEvent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    ModelSettings,
    RawResponsesStreamEvent,
    RunContextWrapper,
    RunItemStreamEvent,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    set_default_openai_client,
    set_default_openai_key,
    trace,
)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_HOST = os.getenv("OPENAI_HOST", "azure")
OPENAI_CHATGPT_MODEL = os.getenv("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-03-01-preview"
TEMPERATURE = 0.0
SEED = 1234


RESPONSE_TOKEN_LIMIT = 512
CHATGPT_TOKEN_LIMIT = 128000

# Env variabl for testing purpose, must be replaced with current date evaluator in future
CURRENT_DATE = "2025-03-01T00:00:00Z" # Start of database
#CURRENT_DATE = "2025-04-16T05:19:54+0000" # Test date 1: 16 April

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

client = AsyncAzureOpenAI(
    api_version="2024-10-21",
    azure_endpoint=f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com",
    azure_ad_token_provider=token_provider,
    azure_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
)

set_default_openai_key(f"{OPENAI_API_KEY}")

@dataclass
class UserInfo:
    auth_header: Optional[dict] = None
    data_type: Optional[str] = None
    date: str = "2025-03-01"  # str(date.today())
    restart: bool = False

class PatientInfo:
    name: str
    age: int
    vaccination_history: List[Dict]

class BookingDetails(BaseModel):
    booking_slot_id: str
    vaccine: str
    clinic: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    google_maps_url: Optional[str] = None


class RescheduleDetails(BaseModel):
    record_id: str  # old
    booking_slot_id: str  # new
    vaccine: str
    previous_clinic: Optional[str] = None
    previous_date: Optional[str] = None
    previous_time: Optional[str] = None
    new_clinic: Optional[str] = None
    new_date: Optional[str] = None
    new_time: Optional[str] = None
    google_maps_url: Optional[str] = None


class CancellationDetails(BaseModel):
    record_id: str
    vaccine: str
    clinic: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class QueryType(Enum):
    MANUAL = "manual"
    SUGGESTED = "suggested"
    FOLLOWUP = "followup"


class PersonaType(str, Enum):
    MYSELF = "myself"
    OTHERS = "others"
    GENERAL = "general"
    UNDEFINED = ""


class PersonaGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNDEFINED = ""


class PersonaAgeType(str, Enum):
    YEARS = "years"
    MONTHS = "months"

@function_tool
async def get_vaccination_history_tool(
    wrapper: RunContextWrapper[UserInfo]
) -> list | str:
    """
    Gets list of past vaccinations of the user and the respective recommended frequencies of vaccines.
    """
    print(f"[TOOL CALL] get_vaccination_history_tool called")
    result = [{"drug_name": "Influenza (INF)", "frequency": "3"},{"drug_name":"HPV", "frequency":"2"}]
    print(f"[TOOL CALL] get_vaccination_history_tool result: {result}")
    return result

@function_tool
async def get_latest_vaccination_tool(
    wrapper: RunContextWrapper[UserInfo], requested_vaccine: str
) -> list | str:
    """
    Gets list of past vaccinations of the user and recommended frequencies for the requested vaccine type, which can be used to inform user about their latest vaccination of the requested type and get confirmation to continue with new booking.

    Args:
      requested_vaccine: User input of vaccine type found from chat history.
    """
    print(f"[TOOL CALL] get_latest_vaccination_tool called with requested_vaccine: {requested_vaccine}")
    result = result = [{"drug_name": "Influenza (INF)", "frequency": "3"}]
    print(f"[TOOL CALL] get_latest_vaccination_tool result: {result}")
    return result


@function_tool
async def recommend_vaccines_tool(wrapper: RunContextWrapper[UserInfo]) -> str:
    """
    Get vaccine recommendations for user based on their demographic.
    """
    print(f"[TOOL CALL] recommend_vaccines_tool called")
    recommendations = [
        {"drug_name": "Influenza (INF)"}]
    print(f"[TOOL CALL] recommend_vaccines_tool result: {recommendations}")
    return json.dumps(recommendations)


@function_tool
async def standardise_vaccine_name_tool(
    wrapper: RunContextWrapper[UserInfo], requested_vaccine: str
) -> dict | str:
    """
    Always use this tool when the step requires it.

    Args:
        requested_vaccine: User input of vaccine type found from chat history.
    """
    print(f"[TOOL CALL] standardise_vaccine_name_tool called with requested_vaccine: {requested_vaccine}")
    standard_name_prompt = f"""
    The input may use informal name, and your task is to map it to the correct official name. For example, the input "flu vaccine" should be mapped to "Influenza (INF)".

    Find the closest match of {requested_vaccine} to the list below:
    - Influenza (INF)
    - Pneumococcal Conjugate (PCV13)
    - Human Papillomavirus (HPV)
    - Tetanus, Diphtheria, Pertussis (Tdap)
    - Hepatitis B (HepB)
    - Measles, Mumps, Rubella (MMR)
    - Varicella (VAR)

    If there is a match, return the value in the list exactly.
    Else, return "Handoff to recommender_agent"

    Official vaccine name:
    """

    messages = build_messages(
        model=OPENAI_CHATGPT_MODEL,
        system_prompt=standard_name_prompt,
        max_tokens=CHATGPT_TOKEN_LIMIT - RESPONSE_TOKEN_LIMIT,
    )

    chat_completion: ChatCompletion = await client.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=messages,
        temperature=0,
        max_tokens=RESPONSE_TOKEN_LIMIT,
        n=1,
        stream=False,
        seed=SEED,
    )

    llm_output = chat_completion.choices[0].message.content
    if llm_output != "None":
        response_dict = {
            "vaccine_name": llm_output,
        }
        print(f"[TOOL CALL] standardise_vaccine_name_tool result: {response_dict}")
        return response_dict
    else:
        print(f"[TOOL CALL] standardise_vaccine_name_tool result: {llm_output}")
        return llm_output


@function_tool
async def get_upcoming_appointments_tool(
    wrapper: RunContextWrapper[UserInfo]
) -> list | str:
    """
    Gets list of current bookings from the user, which can be used to check if user already has an existing booking for the vaccine requested.
    """
    print(f"[TOOL CALL] get_upcoming_appointments_tool called")
    # Get the booked slots from user
    augmented_records = [{"booking_slot_id": "1234567890", "vaccine": "Influenza (INF)", "clinic": "Polyclinic A", "date": "2025-03-01", "time": "10:00 AM"}]
    print(f"[TOOL CALL] get_upcoming_appointments_tool result: {augmented_records}")
    return augmented_records


@function_tool
async def get_clinic_name_response_helper_tool(
    wrapper: RunContextWrapper[UserInfo], clinic_name: str
) -> dict | str:
    """
    Always use this tool when the step requires it.
    Returns polyclinics closest to user's home when 'Not found' given as input.

    Args:
        clinic_name: Takes either the clinic name or 'Not found'
    """
    print(f"[TOOL CALL] get_clinic_name_response_helper_tool called with clinic_name: {clinic_name}")
    # Find polyclinic name not found in chat history, if not found then recommend polyclinics near home
    response_dict = {
        "clinic_name": "Polyclinic A",
    }
    print(f"[TOOL CALL] get_clinic_name_response_helper_tool result: {response_dict}")
    return response_dict


@function_tool
async def get_available_slots_tool(
    wrapper: RunContextWrapper[UserInfo],
    vaccine_name: str,
    clinic: str,
    start_date: str,
    end_date: str,
) -> List[Dict] | str:
    """
    Get available slots for a vaccine at a specific clinic, over a date ranges.

    Args:
        vaccine_name: Official name of vaccine type
        clinic: The name of clinic
        start_date: Start date of date range to search (ISO format)
        end_date: End date of date range to search (ISO format)
    """
    print(f"[TOOL CALL] get_available_slots_tool called with vaccine_name: {vaccine_name}, clinic: {clinic}, start_date: {start_date}, end_date: {end_date}")

    result = [
        {"slot_id": "1234567890", "vaccine": "Influenza (INF)", "clinic": "Polyclinic A", "date": "2025-03-01", "time": "10:00 AM"},
        {"slot_id": "0987654321", "vaccine": "Influenza (INF)", "clinic": "Polyclinic A", "date": "2025-03-02", "time": "11:00 AM"},]
    print(f"[TOOL CALL] get_available_slots_tool result: {result}")
    return json.dumps(result)


@function_tool
async def recommend_gps_tool(
    wrapper: RunContextWrapper[UserInfo]
) -> str:
    """
    Get nearest GPs to user's home address, if there are no available slots found at selected polyclinic.
    """
    recommended_gps = [
        {"name": "GP Clinic A", "address": "123 GP St, Singapore 123456", "phone": "+65 1234 5678"},]
    print(f"[TOOL CALL] recommend_gps_tool result: {recommended_gps}")
    return json.dumps(recommended_gps)




@function_tool
async def new_appointment_tool(
    wrapper: RunContextWrapper[UserInfo], slot_id: str
) -> BookingDetails:
    """
    Handles booking of a new appointment.

    Args:
        slot_id: The 'id' field for slot to be booked
    """
    print(f"[TOOL CALL] new_appointment_tool called with slot_id: {slot_id}")

    response = {"booking_slot_id": slot_id, "vaccine": "Influenza (INF)", "clinic": "Polyclinic A", "date": "2025-03-01", "time": "10:00 AM", "google_maps_url": "https://maps.google.com/?q=Polyclinic+A"}
    print(f"[TOOL CALL] new_appointment_tool result: {response}")
    return json.dumps(response)


@function_tool
async def change_appointment_tool(
    wrapper: RunContextWrapper[UserInfo],
    record_id: str,
    new_slot_id: str,
) -> BookingDetails:
    """
    Handles the changing of location and datetime for existing appointments.

    Args:
        record_id (str): The id for the vaccination appointment record to remove
        new_slot_id (str): The id of slot to reschedule a current slot to
    """
    print(f"[TOOL CALL] change_appointment_tool called with record_id: {record_id}, new_slot_id: {new_slot_id}")

    response = {
        "record_id": record_id,"booking_slot_id": new_slot_id, "vaccine": "Influenza (INF)", "previous_clinic": "Polyclinic A", "previous_date": "2025-03-01", "previous_time": "10:00 AM", "new_clinic": "Polyclinic B", "new_date": "2025-03-02", "new_time": "11:00 AM", "google_maps_url": "https://maps.google.com/?q=Polyclinic+B"}
    print(f"[TOOL CALL] change_appointment_tool result: {response}")
    return json.dumps(response)


@function_tool
async def cancel_appointment_tool(wrapper: RunContextWrapper[UserInfo], record_id: str):
    """
    Handles rescheduling of an existing appointment.
    Args:
        record_id (str): The id for the vaccination appointment record to remove
    """
    print(f"[TOOL CALL] cancel_appointment_tool called with record_id: {record_id}")
 
    response = {
        "record_id": record_id,
        "vaccine": "Influenza (INF)",
        "clinic": "Polyclinic A",
        "date": "2025-03-01",
        "time": "10:00 AM",
    }
    print(f"[TOOL CALL] cancel_appointment_tool result: {response}")
    return json.dumps(response)


vaccination_records_agent = Agent[UserInfo](
    name="vaccination_records_agent",
    instructions=(
        """
    You are tasked with retrieving the records of the user.
    Use get_vaccination_history_tool to retrieve the records.
    Output the results.
    """
    ),
    tools=[get_vaccination_history_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_vaccination_history_tool"),
)


def recommender_agent_prompt(
    context_wrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    context: UserInfo = context_wrapper.context.context
    return f"""
    You are tasked with giving vaccine recommendations for the user based on their vaccination history.
    Today's date: {context.date}
    Follow the steps in order:
    1. Use the recommend_vaccines_tool to get the vaccine recommendations based on their demographic.
    2. Use the get_vaccination_history_tool to retrieve the past vaccination records of the user.
    3. Use the get_upcoming_appointments_tool to retrieve the upcoming appointments user has.
    4. Use the information from the tools and today's date, to help recommend the user which vaccines they should take soon.
    5. Skip this step if the previous agent if you are called by the orchestrator agent. Tell the user to specify clearly one of the vaccines from recommended list for booking.
    """


recommender_agent = Agent[UserInfo](
    name="recommender_agent",
    instructions=recommender_agent_prompt,
    tools=[get_upcoming_appointments_tool, get_vaccination_history_tool, recommend_vaccines_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="recommend_vaccines_tool"),
)


def check_available_slots_agent_prompt(
    context_wrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    context: UserInfo = context_wrapper.context.context
    return f"""
        Follow the steps in order:
        1. See if the user is giving a reply to a question asked by you previously, if not skip this step.
          - If the user chose a slot, handoff to the manage_appointment_agent.
          - If the user requests for slots over other dates or timings, use the get_available_slots_tool again for their new request and skip to step 4.
        2. **Gathering inputs for get_available_slots_tool**: Look at function call result from previous agent and use that as the polyclinic input.
        Look at chat history. If the user specified a date, use that as input to return slots for that date. Else, return slots from {context.date} to 3 days after.
        3. **Get slots from polyclinic**: Use the get_available_slots_tool to find available slots at the polyclinic.
        4. If the tool output returns any available slots, immediately ask the user to choose one of the slots.
        5. Else if the tool output returns no available slots:
            Use the recommend_gps_tool to find and list the nearest GPs to their homes, provide them the link: https://book.health.gov.sg/ for booking with the recommended GPs and immediately handoff to the orchestrator_agent.
        """


check_available_slots_agent = Agent(
    name="check_available_slots_agent",
    instructions=check_available_slots_agent_prompt,
    tools=[get_available_slots_tool, recommend_gps_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_available_slots_tool"),
)


# TODO: Add validity check for polyclinic name
identify_clinic_agent = Agent(
    name="identify_clinic_agent",
    instructions=(
        """
        You are part of a team of agents handling vaccination booking.
        Your task in this team is to only look at chat history and extract from chat history the name of clinic user would like to get.
        Follow the steps in order:
        1. Give the clinic_name you found, or 'Not found' if no clinic name was specified as input to the get_clinic_name_response_helper_tool.
        2. If the tool output returns a list of clinics, immediately ask the user to choose from the list of polyclinics near their home or input another polyclinic and stop.
        3. Handoff to the check_available_slots_agent.
        """
    ),
    handoffs=[check_available_slots_agent],
    tools=[get_clinic_name_response_helper_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_clinic_name_response_helper_tool"),
)


def vaccination_history_check_agent_prompt(
    context_wrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    context: UserInfo = context_wrapper.context.context
    return f"""
        Today's date: {context.date}
        You are part of a team of agents handling vaccination booking.
        Your task in this team is to check if the user is eligible for the vaccine they would like to get based on their vaccination history.
        Follow the steps in order:
        1. Skip this step if you are being called by the recommended_vaccine_check_agent. See if the user is giving a reply to a question asked by you previously.
          - If the reply is affirmative, handoff to identify_clinic_agent. If the reply is not affirmative, then handoff to orchestrator_agent.
        2. Look at the output from the handle_vaccine_names_agent for the vaccine type requested by user.
        3. Use the get_latest_vaccination_tool to get the latest vacccination of user for the vaccine type requested. 
          - If the tool returns a past record, use the record date, today's date and the recommended frequencies of the vaccine inform the user whether or not it is advised for them to proceed with their current new booking. Ask them to decide if they would like to continue the booking.
          - If the tool returns an empty list, then handoff to identify_clinic_agent.
        """


vaccination_history_check_agent = Agent(
    name="vaccination_history_check_agent",
    instructions=vaccination_history_check_agent_prompt,
    handoffs=[identify_clinic_agent],
    tools=[get_latest_vaccination_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_latest_vaccination_tool"),
)


recommended_vaccine_check_agent = Agent(
    name="recommended_vaccine_check_agent",
    instructions=(
        """
        You are part of a team of agents handling vaccination booking.
        Your task in this team is to check if the user is eligible for the vaccine they would like to get.
        Follow the steps in order:
        1. Look at the output from the handle_vaccine_names_agent for the vaccine type requested by user.
        2. Use the recommend_vaccines_tool to get the vaccines that the user should be taking.
          - If the vaccine requested by the user is not within the list of recommended vaccines, tell the user to choose a vaccine from the list instead.
          - If it is, then continue to handoff to the vaccination_history_check_agent.
        """
    ),
    handoffs=[vaccination_history_check_agent],
    tools=[recommend_vaccines_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="recommend_vaccines_tool"),
)


# Changed updated prompt
double_booking_check_agent = Agent(
    name="double_booking_check_agent",
    instructions=(
        """
        You are part of a team of agents handling vaccination booking.
        Your task in this team is to check if the the user is trying to get a vaccine that they already have an existing upcoming appointment for.
        You only have logic about upcoming appointments.
        Follow the steps in order:
        1. Look at the output from the handle_vaccine_names_agent for the vaccine type requested by user.
        2. Use the get_upcoming_appointments_tool to see the appointments the user has previously booked.
          - If the vaccine types of upcoming appointments do not match the requested type, handoff to recommended_vaccine_check_agent.
          - If the vaccine types of those appointments match the requested vaccine type, tell the user about the existing appointment they have for the requested vaccine.
            - If the user wants to make changes to the existing appointment (change date, time or location), handoff to identify_clinic_agent.
            - If the user wants to cancel the existing appointment, handoff to manage_appointment_agent.
            - If the user wants to keep the existing appointment, handoff to orchestrator_agent.
        
        If the user's reply is unrelated to the existing appointment (e.g. ask about side effects), handoff to the interrupt_handler_agent.
        """
    ),
    handoffs=[recommended_vaccine_check_agent, identify_clinic_agent],
    tools=[get_upcoming_appointments_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_upcoming_appointments_tool"),
)


handle_vaccine_names_agent = Agent(
    name="handle_vaccine_names_agent",
    instructions=(
        "You are part of a team of agents handling vaccination booking."
        "Your task in this team is to only look at chat history and extract from chat history the vaccine type user would like to get."
        "Follow these steps in order:"
        "1. Find the vaccine name mentioned and Use the standardise_vaccine_name_tool to convert it to the official name."
        "2. If the tool output is a vaccine name, use its output and handoff to the double_booking_check_agent."
        "3. If the tool output asks to handoff to recommender_agent, handoff to the recommender_agent."
    ),
    handoffs=[double_booking_check_agent, recommender_agent],
    tools=[standardise_vaccine_name_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="standardise_vaccine_name_tool"),
)


manage_appointment_agent = Agent(
    name="manage_appointment_agent",
    instructions=(
        "Your task is to complete actions the user would like regarding vaccination appointments."
        "For new bookings, use the new_appointment_tool"
        "For making changes to existing bookings, use the change_appointment_tool."
        "For cancellations, get the record_id of the existing appointment from upcoming_appointment_agent output. Then use the cancel_appointment_tool."
        "Return the output from any toolcall."
    ),
    tools=[new_appointment_tool, change_appointment_tool, cancel_appointment_tool],
    model="gpt-4o-mini",
)


modify_existing_appointment_agent = Agent(
    name="modify_existing_appointment_agent",
    instructions=(
        """
    You are part of a team of agents handling modification of existing vaccination appointments.
    Your task in this team is to check which upcoming vaccination appointment the user would like to modify.
    Follow the steps in order:
    1. See if the user is giving a reply to a question asked by you previously, or if the user's requested upcoming vaccination appointment is found. If not, skip this step.
        - If an upcoming vaccination appointment is selected by the user or found by you:
            - If the user wants to make changes to it, for example, change date or location, handoff to identify_clinic_agent.
            - If the user wants to cancel it, handoff to manage_appointment_agent.
    2. Else, use the get_upcoming_appointments_tool to get the list of upcoming appointments. If you find any, display the polyclinic name, date and time and vaccine type.
    3. Ask the user to select which upcoming appointment to modify, and clarify their intent to cancel or make changes (date, time or lovation) if they have not specified. 
    """
    ),
    tools=[get_upcoming_appointments_tool],
    handoffs=[identify_clinic_agent, manage_appointment_agent],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_upcoming_appointments_tool"),
)


appointments_agent = Agent[UserInfo](
    name="appointments_agent",
    instructions=(
        """
    # System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n

    Your task is to decide which agent to handoff to, do not ask user anything.
    If the user wants to book a new slot, handoff to handle_vaccine_names_agent.
    If the user is asking about rescheduling or cancelling an existing booking, handoff to modify_existing_appointment_agent.
    """
    ),
    handoffs=[
        handle_vaccine_names_agent,  # starts flow to get location and vaccine name.
        modify_existing_appointment_agent,
    ],
    model="gpt-4o-mini",
)


orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a helpful assistant that directs user queries to the appropriate agents."
        "Take the conversation history as context when deciding who to handoff to next."
        "If they want to book vaccination appointments, but did not mention which vaccine they want, handoff to the recommender_agent."
        "If they mentioned their desired vaccine and would like to book an appointment, handoff to appointments_agent."
        "If they ask for vaccination reccomendations, handoff to recommender_agent."
        "If they ask about vaccination records, like asking about their past vaccinations, handoff to vaccination_records_agent."
        "Otherwise, handoff to general_questions_agent."
    ),
    handoffs=[
        appointments_agent,
        recommender_agent,
        vaccination_records_agent,
   
    ],
    model="gpt-4o-mini",
)


# Add backlinks from sub-agents
check_available_slots_agent.handoffs.append(manage_appointment_agent)
double_booking_check_agent.handoffs.append(manage_appointment_agent)
double_booking_check_agent.handoffs.append(orchestrator_agent)
vaccination_history_check_agent.handoffs.append(orchestrator_agent)
check_available_slots_agent.handoffs.append(orchestrator_agent)

# -- Define Sub-Servers --
booking_mcp = FastMCP(name="BookingService")

AGENT_MAP = {
    "modify_existing_appointment_agent": modify_existing_appointment_agent,
    "appointments_agent": appointments_agent,
    "handle_vaccine_names_agent": handle_vaccine_names_agent,
    "double_booking_check_agent": double_booking_check_agent,
    "recommender_agent": recommender_agent,
    "vaccination_history_check_agent": vaccination_history_check_agent,
    "identify_clinic_agent": identify_clinic_agent,
    "check_available_slots_agent": check_available_slots_agent,
    "manage_appointment_agent": manage_appointment_agent,
    "orchestrator_agent": orchestrator_agent,
    "vaccination_records_agent": vaccination_records_agent,
    "recommended_vaccine_check_agent": recommended_vaccine_check_agent
}
# -- BookingService --

@booking_mcp.tool()
async def process_booking_request(
    user_query: str,
    context: str,
    user_id: str,
    user_name: Optional[str] = None,
    last_agent_name: Optional[str] = None,
    
) -> Dict[str, Any]:
    """
    Processes a user's query related to appointments. This can include viewing past bookings,
    or initiating a process to modify/cancel an existing booking.
    Args:
        user_query: The user's natural language query related to appointments.
        context: Any context information and past conversation/chat history that is relevant to current the user_query
        user_id: The unique identifier for the user.
        last_agent_name: The name of the last agent that handled the request. This is so that that last agent can be set as the starting agent for the this run.
        user_name: (Optional) The name of the user.
    """
    print(f"[FastMCP Tool] 'process_booking_request' called for user_id: {user_id} with query: '{user_query}'")
    try:
        wrapper = RunContextWrapper(
            context=UserInfo(
                auth_header={
                    "Authorization": f"Bearer XXX",
                    "Content-Type": "application/json",
                }
            )
        )
        # The starting agent is now the more general 'appointments_agent'
        print(f"Last agent name: {last_agent_name}")
        starting_agent = AGENT_MAP.get(last_agent_name, orchestrator_agent)
        print(f"Calling await Runner.run with starting_agent: '{starting_agent.name}', user_query: '{user_query}',context/chathistory: '{context}', context for user_id: {user_id}")
        run_result = await Runner.run(
            starting_agent=starting_agent, # Entry point for booking service
            input= context +user_query ,
            context=wrapper,
            max_turns=10
        )
        last_agent_name = run_result.last_agent.name
        print(f"Last agent in the run: {last_agent_name}")
        print(f"AppointmentsAgent (and potential handoffs) Runner.run call completed. Final output: {run_result.final_output}")
        return {
            "user_query": user_query,
            "agent_final_response": run_result.final_output,
            "status": "success",
            "last_agent_name": last_agent_name,
        }
    
    except Exception as e:
        error_msg = f"An unexpected error in 'process_booking_request': {e}"
        return {"error": error_msg, "status": "tool_exception"}
    

# --- Define Main MCP Server ---
mcp = FastMCP(name="MCPServer")

# --- Setup to import subservers and configure OpenAI ---
async def setup():
    print("Initialising setup for MCPServer...")

    try:
        client = AsyncOpenAI()
        set_default_openai_client(client)
        print("Default OpenAI client configured successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to configure OpenAI client: {e}.")
       
    # Import booking service with prefix "booking_agent"
    await mcp.import_server("appointments", booking_mcp)
    print("BookingAppointmentService (appointment_booking_mcp) imported with prefix 'appointments_agent'.")

    # Import general info service with prefix "general_info_agent"
    #await mcp.import_server("general", general_info_mcp)
    print("GeneralInfoService (general_info_mcp) imported with prefix 'general_info_agent'.")

    print("MCPServer setup completed.")


if __name__ == "__main__":
    print("Starting FastMCP server...")
    asyncio.run(setup())

    print(f"FastMCP server '{mcp.name}' is preparing to run.")
    mcp.run(transport="sse")