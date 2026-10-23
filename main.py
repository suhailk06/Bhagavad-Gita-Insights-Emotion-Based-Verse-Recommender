from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Bhagavad Gita Insights")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request:Request):
    return templates.TemplateResponse({"request":request},"home.html")
