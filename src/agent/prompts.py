
# chunk the prompt to construct in one whole prompt inside the harness


def main_goal_prompt() -> str:
    return """Inspect the codebase to create a proper database schema report such as sql query (raw and ORM), constraints, CTE, database, etc...

Result:
Use the standard markdown format (headers, tables, sections, link, separators, etc...). Use the create_audit_report tool.
Header: Title, date, writer (Agent)
Overview: 1-2 paragraphs
Sections: 
- Create your own sections according to your report.
- Prerequisites: table schema, ERD diagram
"""


def constraint_prompt() -> str:
    return """
Constraints:
Do NOT read the whole codebase one file at a time. Use the available tools to find the relevant files for the goal.
"""
   
    
def workflow_prompt() -> str:
    return """
Workflow and Best Practices:
1. System: Pass a path
2. Agent: Use the list_folder_content tool to analyze and separate schema files (schema.sql, migrations/, ORM) from app code (repos/, services/, routes/).
3. Agent: Use the search_symbol tool to find the functions, sql or orm query, migrations.
4. 

"""

    
    
    
    
    
    
    




