import os
from pydevs.types.completion import Message
from pydevs.services.openai import OpenAIService
from prompts import format_note_system_message, refine_note_system_message

note_types = [
    {
        "name": "tasks",
        "description": "Tasks to be done",
        "formatting": "bullet list in which each item start with the project name followed by a concise description of the task",
        "context": """Projects:
        
        - Tech•sistence: A newsletter about the latest tech and workflow tools; coworker: Greg
        - eduweb.pl: An online platform with courses for programmers & designers; coworkers: Greg, Peter (video), Joanna
        - easy.tools: A platform for selling digital products and managing online business; coworkers: Greg, Peter (dev), Marta, Michał
        - heyalice.app: Desktop LLM client for macOS; coworker: Greg""",
    },
    {
        "name": "grocery-list",
        "description": "Grocery list",
        "formatting": "bullet list of items to buy, optionally grouped by categories",
        "context": """# Grocery List Formatting Guide

1. Structure:
   - Begin with a clear title: "Grocery List for [Date/Week]"
   - Organize items by store layout or category (e.g., Produce, Dairy, Meats)

2. Item Format:
   - List each item on a new line
   - Include quantity and unit (e.g., "2 lbs chicken breast")
   - Add brief descriptors if necessary (e.g., "ripe avocados")

3. Categorization:
   - Group similar items under headers
   - Use bold formatting for category headers

4. Prioritization:
   - Place essential items at the top of each category
   - Mark urgent items with an asterisk (*)

5. Flexibility:
   - Include space for spontaneous additions
   - Note potential substitutions in parentheses

6. Meal Planning Integration:
   - Link items to planned meals where applicable
   - Use a coding system (e.g., [M1] for Meal 1) to connect items to specific recipes

7. Budget Considerations:
   - Include estimated prices if budget tracking is important
   - Mark sale items or those with coupons

8. Special Notes:
   - Add any dietary restrictions or preferences at the top
   - Include a section for non-food items if necessary

9. Digital Enhancements:
   - If converting to a digital format, include clickable checkboxes
   - Add links to recipes or nutritional information where relevant

Remember to keep the list concise, clear, and tailored to the user's specific needs and shopping habits.""",
    },
    {
        "name": "meeting-notes",
        "description": "Meeting notes",
        "formatting": "structured notes with headers for agenda, decisions, action items, and follow-ups",
        "context": """Projects:
        
        - Tech•sistence: A newsletter about the latest tech and workflow tools; coworker: Greg
        - eduweb.pl: An online platform with courses for programmers & designers; coworkers: Greg, Peter (video), Joanna
        - easy.tools: A platform for selling digital products and managing online business; coworkers: Greg, Peter (dev), Marta, Michał
        - heyalice.app: Desktop LLM client for macOS; coworker: Greg""",
    },
]

def determine_note_type(openai_service:OpenAIService, message: str) -> str:
    """Determine the type of note based on the message content."""
    
    system_content = "You are an intelligent note classification system. Your task is to categorize the given message into one of the following note types:\n\n"
    system_content += "\n".join([f"- {type['name']}: {type['description']}" for type in note_types])
    system_content += "\n\nAnalyze the content and context of the message carefully. Respond ONLY with the lowercased name of the most appropriate note type. If uncertain, choose the closest match."
    
    response = openai_service.text_completion(
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": message}
        ],
        model="gpt-4o"
    )
    
    return response[0]["content"].strip().lower() if response else "unknown"

def format_note(openai_service:OpenAIService, type_name: str, message: str) -> str:
    """Format the message according to the specified note type."""
    
    note_type = next((t for t in note_types if t["name"] == type_name), None)
    if not note_type:
        return f'Error: Invalid note type "{type_name}"'
    
    response = openai_service.text_completion(
        messages=[
            {"role": "system", "content": format_note_system_message(openai_service, note_type)},
            {"role": "user", "content": message}
        ],
        model="gpt-4o"
    )
    
    return response[0]["content"] if response else f'There was an error formatting the note. Original message: {message}'

def refine_note(openai_service:OpenAIService, note: str, original_message: str) -> str:
    """Refine the formatted note."""
    
    response = openai_service.text_completion(
        messages=[
            {"role": "system", "content": refine_note_system_message(openai_service, note, original_message)},
            {"role": "user", "content": "Please refine this note. Write refined version and nothing else."}
        ],
        model="gpt-4o"
    )
    
    return response[0]["content"] if response else note

def process_note(message: str, openai_service:OpenAIService) -> None:
    """Process a note through the complete pipeline."""
    # Create reasoning.md in the same directory as this script
    reasoning_path = os.path.join(os.path.dirname(__file__), 'reasoning.md')
    
    print("Clear the file")
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        f.write('')
    
    print("tep 1: Determine note type")
    note_type = determine_note_type(openai_service, message)
    with open(reasoning_path, 'a', encoding='utf-8') as f:
        f.write(f"Note Type: {note_type}\n\n---\n\n")
    
    print("Step 2: Format note")
    formatted_note = format_note(openai_service, note_type, message)
    with open(reasoning_path, 'a', encoding='utf-8') as f:
        f.write(f"Formatted Note:\n{formatted_note}\n\n---\n\n")
    
    print("Step 3: Refine note")
    refined_note = refine_note(openai_service, formatted_note, message)
    with open(reasoning_path, 'a', encoding='utf-8') as f:
        f.write(f"Refined Note:\n{refined_note}\n\n")

if __name__ == "__main__":
    # Example message
    message = """So i spoke with Greg about the latest features that should include voice interaction in the real time you know, but we also need a good presentation for the next conference about this app."""
    openai_service = OpenAIService()
    # Process the note
    process_note(message, openai_service)