# Citi Project

A Python project with MCP (Model Context Protocol) server implementation and API logic components.

## Project Structure

```
Citi/
├── API_Logics/          # API logic components
│   ├── coder.py
│   ├── giter.py
│   └── mcp.py
├── MCP_Server/          # MCP Server implementation
│   ├── Database_Server.py
│   ├── Evaluation_Server.py
│   ├── main.py
│   ├── mcp_mounter.py
│   └── pyproject.toml
├── requirements.txt     # Project dependencies
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- Node.js (for MCP Inspector)
- UV package manager (recommended)

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Citi
```

### 2. Set up virtual environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# For Linux/Mac
# source venv/bin/activate
```

### 3. Install dependencies
```powershell
# Install base requirements
pip install -r requirements.txt

# Install additional packages with UV
uv add fastapi fastmcp sqlalchemy python-dotenv sqlparse unicorn[standard] aiomysql aiohttp mcp langgraph
```

## Usage

### Starting the MCP Server
```powershell
# Navigate to MCP_Server directory
cd MCP_Server

# Run the MCP server
uv run mcp_mounter.py
```

### Using the MCP Inspector
In a separate terminal window:
```bash
npx @modelcontextprotocol/inspector
```
### Starting the MCP Client
```powershell
# Navigate to MCP_Client directory
cd MCP_Client

# Run the MCP server
fastapi run client.py
```




