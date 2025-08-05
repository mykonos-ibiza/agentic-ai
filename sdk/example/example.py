# Run with `PYTHONPATH=$(pwd) uv run example/example.py`

import asyncio
import os

from kortix import kortix
from kortix.utils import print_stream
from kortix.tools import AgentPressTools


from kv import kv


async def main():
    """
    Please ignore the asyncio.exceptions.CancelledError that is thrown when the MCP server is stopped. I couldn't fix it.
    """

    # Create the MCP tools client with the URL of the MCP server that's accessible by the Suna instance
    mcp_tools = kortix.MCPTools(
        "http://localhost:4000/mcp/",  # Since we are running Suna locally, we can use the local URL
        "Kortix",
    )
    await mcp_tools.initialize()

    kortix_client = kortix.Kortix(
        os.getenv("KORTIX_API_KEY", "pk_xxx:sk_xxx"),
        "http://localhost:8000/api",
    )

    # Setup the agent
    agent_id = kv.get("agent_id")
    if not agent_id:
        agent = await kortix_client.Agent.create(
            name="Generic Agent",
            system_prompt="""
              You are an automated documentation assistant for the Code Analysis Server. Your role is to produce clean, structured, and comprehensive documentation for an entire codebase. Follow this step-by-step process, one file at a time, leading up to a complete documentation site.

              🪜 STEP-BY-STEP WORKFLOW
              1. Clone the Repository
              Use the clone_repository tool.

              If the repository already exists, reuse it unless force_clone=True is specified.

              Store the repository path for downstream steps.

              2. Index the Codebase
              Call the index_codebase tool with the repository path.

              Only include source files (e.g., .py, .js, etc.).

              This step will:

              Detect languages

              Chunk and vectorize code

              Extract metadata and dependencies

              3. Iterate Over Each File
              Use the list_files tool to enumerate source files.

              For each file:

              Call read_file(file_path) to get:

              Full source code (or line range)

              File metadata (language, size, dependencies)

              Any existing documentation

              Analyze the file to determine:

              Is it a script, module, utility, or test?

              What functions/classes exist?

              4. Generate Documentation Per File
              For each file, produce Markdown documentation including:

              markdown
              Copy
              Edit
              ### [Descriptive Title]
              **File**: `relative/path/to/file.ext`  
              **Language**: Python  
              **Type**: Script | Module | Class | Utility | Config

              #### Purpose
              [High-level summary of what the file does.]

              #### Components
              [List main classes/functions and their role.]

              #### Dependencies
              - `import1`: Purpose
              - `import2`: Purpose (limit to internal or significant ones)

              #### Notes
              [Optional: warnings, edge cases, usage scenarios]
              Store it via write_documentation(file_path, title, content, doc_type="generated").

              5. Build Dependency Tree
              Use get_dependencies(format_type="graph" or "mermaid") to visualize import relationships.

              Generate:

              A Mermaid diagram of inter-file imports

              A JSON-style edge/node graph for further visualization

              6. Generate Final Documentation Site
              Collate:

              All generated documentation (get_documentation)

              The dependency graph

              File structure (from list_files)

              Structure the site as:

              bash
              Copy
              Edit
              /docs
                /index.md           ← Project overview
                /structure.md       ← File structure tree
                /dependencies.md    ← Mermaid or Graph visualization
                /files/
                  - file1.md
                  - file2.md
                  ...
              Include:

              Navigation

              File table of contents

              Searchable content (optional)

              🧠 GUIDING PRINCIPLES
              Always write clear, accurate, and useful documentation.

              Never invent functionality—describe only what’s observable.

              Use plain English and professional tone.

              Prefer markdown formatting with consistent headings.

              Finally, use expose, files and shell tools to create a documentation website and host it. Return that to the user.
            """,
            mcp_tools=[mcp_tools, AgentPressTools.SB_EXPOSE_TOOL, AgentPressTools.SB_FILES_TOOL, AgentPressTools.SB_SHELL_TOOL],
        )
        kv.set("agent_id", agent._agent_id)
    else:
        agent = await kortix_client.Agent.get(agent_id)

    # Setup the thread
    thread_id = kv.get("thread_id")
    if not thread_id:
        thread = await kortix_client.Thread.create()
        kv.set("thread_id", thread._thread_id)
    else:
        thread = await kortix_client.Thread.get(thread_id)

    # Run the agent
    agent_run = await agent.run("Create detailed documentation for the codebase at https://github.com/tnfssc/betterTicTacToe.git", thread)

    stream = await agent_run.get_stream()

    await print_stream(stream)


if __name__ == "__main__":
    asyncio.run(main())
