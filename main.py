import streamlit as st
import pandas as pd
import os
import json
import re
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import autogen

load_dotenv()

# Google Gemini Model Configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Autogen LLM Config
config_list = [
    {
        "model": "gemini-1.5-flash", # Using a suitable model for the task
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "api_type": "google"
    }
]

llm_config = {
    "config_list": config_list,
    "seed": 42,
    "temperature": 0.7, # Added temperature for creativity
    "config_list": config_list, # Ensure config_list is present
}

# Autogen Agents
researcher_agent = autogen.AssistantAgent(
    name="Researcher",
    llm_config=llm_config,
    system_message="""You are a Blog Researcher whose SOLE purpose is to gather and structure information for a blog post on a given topic.
    You will receive the blog post topic.
    Your task is to perform research and provide key points, facts, and potential sections directly relevant to THIS specific topic.
    Provide the research findings in a clear, structured format (e.g., outline, bullet points, sections), suitable for a writer to use.
    **EXTREMELY IMPORTANT: YOUR ONLY OUTPUT MUST BE THE RESEARCH FINDINGS FOR THE PROVIDED TOPIC. DO NOT include any conversational text, questions, greetings, apologies, or internal system messages like 'TERMINATE' or references to future steps or other agents. Do not discuss processes like fact-checking or visual integration. Provide your complete, topic-specific research findings in a single, direct response.**
    """
)

writer_agent = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""You are a Blog Writer.
    You will receive research data from the researcher.
    Your task is to write a comprehensive, engaging, and well-structured blog post based on the provided research.
    Format the output as the full blog post content.
    **IMPORTANT: ONLY output the blog post content. Do not include any conversational text, greetings, or internal system messages like 'TERMINATE'. Provide the complete blog post in a single response.**
    """
)

# Validation Agent (AutoGen Agent with Google Gemini)
editor_agent = autogen.AssistantAgent(
    name="Editor",
    llm_config=llm_config,
    system_message="""
    You are a blog post editor expert.
    You will receive a blog post draft.
    Your task is to validate and improve the content, making it more meaningful, clear, and engaging.
    Check for flow, grammar, spelling, and overall quality.
    Edit the content as necessary and provide the final, polished blog post.
    **IMPORTANT: ONLY output the final, polished blog post. Do not include any conversational text, greetings, or internal system messages like 'TERMINATE'. Provide the complete, final blog post in a single response.**
    """
)

# User Proxy Agent for Orchestration
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER", # Set to NEVER for autonomous execution in Streamlit
    max_consecutive_auto_reply=2, # This is the max replies the proxy can get in a single *initiated* chat
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"), # Keep termination check just in case
    code_execution_config=False, # No code execution needed for this task
    llm_config=llm_config, # User proxy can also use LLM if needed for internal logic, but its system message restricts its chat behavior
    system_message="""You are a silent and efficient orchestrator. Your SOLE purpose is to initiate tasks with other agents and relay their final, cleaned outputs sequentially through the workflow. DO NOT generate ANY conversational responses, ask agents questions, guide their process, acknowledge their messages, or include any text in the chat other than initiating the specific task for the next agent using the previous agent's output. Just initiate the task and capture the final output for the next step.""" # Much stricter system message
)

# Function to clean agent output from potential internal messages
def clean_agent_output(output):
    """Removes common Autogen internal messages from the start/end of agent output."""
    if not isinstance(output, str):
        return output
    # Remove lines starting with '>>>>>>>>' and potentially ending with 'TERMINATING RUN'
    # Use regex with DOTALL to match across lines
    output = re.sub(r'^\s*>>>>>>>> .*?\n', '', output, flags=re.DOTALL)
    output = re.sub(r'\n>>>>>>>> .*?\s*$', '', output, flags=re.DOTALL)
     # Remove any leading/trailing whitespace
    return output.strip()


# Streamlit App
st.title("BlogBot")

blog_name = st.text_input("Enter a topic name for the blog post:")
if st.button("Generate Blog Post"):
    if blog_name:
        with st.spinner("Generating blog post..."):

            # --- Orchestration Flow ---

            # Step 1: Researcher gathers information
            # st.info("Researcher is gathering information...")
            research_task = f"Research content for a blog post on the topic: {blog_name}. Provide key points, facts, and potential sections."
            # Allow up to 5 auto-replies for research for robustness
            research_result = user_proxy.initiate_chat(
                researcher_agent,
                message=research_task,
                clear_history=True, # Start a fresh conversation
                max_consecutive_auto_reply=1 # Allow a few turns
            )
            # Extract and clean the research content
            research_content = research_result.summary if research_result and research_result.summary else "No research content generated."
            research_content = clean_agent_output(research_content) # Clean the output

            # Debug: Show the captured research content
            # st.write("--- Debug: Research Content Captured ---")
            # st.write(research_content)
            # st.write("------------------------------------")


            # st.subheader("Research Findings:")
            # st.write(research_content)

            # Check if research content looks like valid content
            if not research_content or "No research content generated." in research_content or research_content.strip().lower().startswith("okay") or research_content.strip().lower().startswith("awaiting") or "fact-checking process" in research_content.lower() or "visual integration" in research_content.lower() or research_content.strip().lower().startswith("i will now proceed"):
                 st.error("Research failed or returned invalid content. Cannot proceed with writing.")
            else:
                # Step 2: Writer drafts the blog post based on research
                st.info("Writer is drafting the blog post...")
                writing_task = f"Write a comprehensive blog post based on the following research findings:\n\n{research_content}\n\nEnsure it is engaging and well-structured."
                 # Allow up to 5 auto-replies for writing
                writing_result = user_proxy.initiate_chat(
                    writer_agent,
                    message=writing_task,
                    clear_history=True, # Start a fresh conversation
                    max_consecutive_auto_reply=1
                )
                # Extract and clean the blog post draft
                blog_post_draft = writing_result.summary if writing_result and writing_result.summary else "No blog post draft generated."
                blog_post_draft = clean_agent_output(blog_post_draft) # Clean the output

                # Debug: Show the captured blog post draft
                # st.write("--- Debug: Blog Post Draft Captured ---")
                # st.write(blog_post_draft)
                # st.write("------------------------------------")

                # st.subheader("Blog Post Draft:")
                # st.write(blog_post_draft)

                if not blog_post_draft or "No blog post draft generated." in blog_post_draft or blog_post_draft.strip().lower().startswith("i cannot create") or blog_post_draft.strip().lower().startswith("i will now proceed"): # Added check for previous error message
                    st.error("Writing failed or returned no content. Cannot proceed with editing.")
                else:
                    # Step 3: Editor reviews and improves the blog post
                    st.info("Editor is reviewing and improving the blog post...")
                    editing_task = f"Review and improve the following blog post for clarity, flow, and accuracy. Make it more meaningful and engaging. Provide the final, polished blog post.\n\n{blog_post_draft}"
                     # Allow up to 5 auto-replies for editing
                    editing_result = user_proxy.initiate_chat(
                        editor_agent,
                        message=editing_task,
                        clear_history=True, # Start a fresh conversation
                        max_consecutive_auto_reply=1
                    )
                    # Extract and clean the final blog post
                    final_blog_post = editing_result.summary if editing_result and editing_result.summary else "No final blog post generated."
                    final_blog_post = clean_agent_output(final_blog_post) # Clean the output
                    st.subheader("Final Blog Post:")
                    st.write(final_blog_post)

                    if final_blog_post and "No final blog post generated." not in final_blog_post and not final_blog_post.strip().lower().startswith("i will now proceed"):
                         st.success("Blog post generation complete!")
                    else:
                         st.error("Editing failed or returned no content.")

    else:
        st.warning("Please enter a topic name.")
