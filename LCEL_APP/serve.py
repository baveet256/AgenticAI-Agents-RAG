from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv()
import os 

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Gemma2-9b-It")

from langchain_core.prompts import ChatPromptTemplate

general_prompt = "Translate the following to {language}."

prompt = ChatPromptTemplate.from_messages(
    ## list of tuples
    [("system",general_prompt),("user","{text}")]
)
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

chain = prompt | llm | parser


app = FastAPI()

from fastapi import Request
from pydantic import BaseModel

class InputPayload(BaseModel):
    language: str
    text: str

@app.post("/chain/invoke")
async def invoke_chain(payload: InputPayload):
    result = chain.invoke({"language": payload.language, "text": payload.text})
    return {"output": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
    