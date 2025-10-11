import contextlib
import uvicorn
from fastapi import FastAPI
from Evaluation_Server import mcp as evaluate_mcp
from Database_Server import mcp as db_mcp
from Simple_Server import mcp as math_mcp
from Mongo_Server import mcp as mongo_mcp
import os


# Create a combined lifespan to manage both session managers
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        # add mcps here
        await stack.enter_async_context(evaluate_mcp.session_manager.run())
        await stack.enter_async_context(db_mcp.session_manager.run())
        await stack.enter_async_context(math_mcp.session_manager.run())
        await stack.enter_async_context(mongo_mcp.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)

# Mount the MCP apps
app.mount("/evaluate", evaluate_mcp.streamable_http_app())
app.mount("/db", db_mcp.streamable_http_app())
app.mount("/math", math_mcp.streamable_http_app())
app.mount("/mongo", mongo_mcp.streamable_http_app())

PORT = os.environ.get("PORT", 10000)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)